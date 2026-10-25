package main

import (
	"context"
	"encoding/json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"io"
	"net/http"
	"paperless-gpt/internal/llm"
	"testing"
)

func TestApplyOllamaMetadataEnvironment(t *testing.T) {
	// A non-empty starting point, so "invalid input is ignored" can be told
	// apart from "invalid input resets to zero".
	baseline := func() OllamaMetadataConfig {
		return OllamaMetadataConfig{
			Defaults: OllamaGenerationSettings{
				Temperature:   floatPointer(0.1),
				MaxTokens:     intPointer(20),
				ContextLength: intPointer(512),
				KeepAlive:     stringPointer("5m"),
				Think:         json.RawMessage("true"),
			},
		}
	}

	t.Run("invalid values are ignored rather than applied", func(t *testing.T) {
		t.Setenv("LLM_TEMPERATURE", "NaN")
		t.Setenv("LLM_MAX_TOKENS", "0")
		t.Setenv("OLLAMA_CONTEXT_LENGTH", "-1")
		t.Setenv("OLLAMA_KEEP_ALIVE", "yesterday")
		t.Setenv("OLLAMA_THINK", "maximum")

		current := baseline()
		applyOllamaMetadataEnvironment(&current)
		assert.Equal(t, 0.1, *current.Defaults.Temperature)
		assert.Equal(t, 20, *current.Defaults.MaxTokens)
		assert.Equal(t, 512, *current.Defaults.ContextLength)
		assert.Equal(t, "5m", *current.Defaults.KeepAlive)
		assert.JSONEq(t, "true", string(current.Defaults.Think))
	})

	t.Run("valid values overlay the defaults", func(t *testing.T) {
		t.Setenv("LLM_TEMPERATURE", "0")
		t.Setenv("LLM_MAX_TOKENS", "-1")
		t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
		t.Setenv("OLLAMA_KEEP_ALIVE", "0")
		t.Setenv("OLLAMA_THINK", "high")

		current := baseline()
		applyOllamaMetadataEnvironment(&current)
		// Explicit zero and -1 have to survive: they are meaningful values
		// here, not "unset".
		assert.Equal(t, 0.0, *current.Defaults.Temperature)
		assert.Equal(t, -1, *current.Defaults.MaxTokens)
		assert.Equal(t, 4096, *current.Defaults.ContextLength)
		assert.Equal(t, "0", *current.Defaults.KeepAlive)
		assert.JSONEq(t, `"high"`, string(current.Defaults.Think))
	})
}

func TestCreateLLMOllamaPreservesNativeSettings(t *testing.T) {
	var wire map[string]any
	s := testOllamaServer(t, func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&wire))
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"answer"},"done":true,"done_reason":"stop"}`)
	})
	t.Setenv("OLLAMA_HOST", s.URL)
	t.Setenv("OLLAMA_THINK", "false")
	t.Setenv("OLLAMA_KEEP_ALIVE", "5m")
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "4096")
	t.Setenv("LLM_MAX_TOKENS", "30")
	t.Setenv("LLM_TEMPERATURE", "0")
	prevProvider, prevModel := llmProvider, llmModel
	llmProvider, llmModel = "ollama", "test"
	t.Cleanup(func() { llmProvider, llmModel = prevProvider, prevModel })
	m, err := createLLM()
	require.NoError(t, err)
	_, err = m.Generate(context.Background(), llm.Text("task"))
	require.NoError(t, err)
	assert.Equal(t, false, wire["think"])
	assert.Equal(t, "5m0s", wire["keep_alive"])
	opts := wire["options"].(map[string]any)
	assert.Equal(t, float64(0), opts["temperature"])
	assert.Equal(t, float64(30), opts["num_predict"])
	assert.Equal(t, float64(4096), opts["num_ctx"])
	for _, key := range []string{"max_tokens", "keep_alive", "think", "format"} {
		assert.NotContains(t, opts, key)
	}
}
