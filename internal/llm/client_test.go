package llm

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/eino-contrib/jsonschema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

type modelFunc func(context.Context, []*schema.Message, ...model.Option) (*schema.Message, error)

func (f modelFunc) Generate(c context.Context, m []*schema.Message, o ...model.Option) (*schema.Message, error) {
	return f(c, m, o...)
}
func response(args string) *schema.Message {
	return &schema.Message{ToolCalls: []schema.ToolCall{{Type: "function", Function: schema.FunctionCall{Name: ResultTool, Arguments: args}}}}
}
func TestStructuredResultRejectsIncompleteAndUnrequestedValues(t *testing.T) {
	spec := Object(map[string]*jsonschema.Schema{"text": {Type: "string"}})
	for _, v := range []*schema.Message{nil, {Content: "okay"}, response(`{}`), response(`{"text":1}`), response(`{"text":"ok","other":1}`), {ResponseMeta: &schema.ResponseMeta{FinishReason: "length"}}, response(`{"text":`)} {
		_, err := Generate(context.Background(), modelFunc(func(context.Context, []*schema.Message, ...model.Option) (*schema.Message, error) { return v, nil }), Text("task"), spec)
		require.Error(t, err)
	}
	for _, args := range []string{`{"text":""}`, `{"text":"recognized"}`} {
		out, err := Generate(context.Background(), modelFunc(func(context.Context, []*schema.Message, ...model.Option) (*schema.Message, error) {
			return response(args), nil
		}), Text("task"), spec)
		require.NoError(t, err)
		assert.JSONEq(t, args, out.Content)
	}
}
func TestEinoImageToolAndThinkingWire(t *testing.T) {
	var wire map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&wire))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"r","type":"function","function":{"name":"submit_document_result","arguments":"{\"text\":\"page\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}`))
	}))
	defer server.Close()
	m, err := New(context.Background(), Config{Provider: "openai", Model: "deepseek-v4.1-flash", APIKey: "key", BaseURL: server.URL})
	require.NoError(t, err)
	b64 := "aW1hZ2U="
	in := []*schema.Message{{Role: schema.User, UserInputMultiContent: []schema.MessageInputPart{{Type: schema.ChatMessagePartTypeText, Text: "Read this page"}, {Type: schema.ChatMessagePartTypeImageURL, Image: &schema.MessageInputImage{MessagePartCommon: schema.MessagePartCommon{Base64Data: &b64, MIMEType: "image/png"}}}}}}
	out, err := Generate(context.Background(), m, in, Object(map[string]*jsonschema.Schema{"text": {Type: "string"}}))
	require.NoError(t, err)
	assert.JSONEq(t, `{"text":"page"}`, out.Content)
	assert.Equal(t, 110, Usage(out)["TotalTokens"])
	assert.Equal(t, float64(0), wire["temperature"])
	assert.Equal(t, "disabled", wire["thinking"].(map[string]any)["type"])
	assert.Equal(t, ResultTool, wire["tool_choice"].(map[string]any)["function"].(map[string]any)["name"])
	msg := wire["messages"].([]any)[0].(map[string]any)
	assert.Equal(t, "data:image/png;base64,"+b64, msg["content"].([]any)[1].(map[string]any)["image_url"].(map[string]any)["url"])
}

func TestOutputLimitPreservesUsage(t *testing.T) {
	m := modelFunc(func(context.Context, []*schema.Message, ...model.Option) (*schema.Message, error) {
		return &schema.Message{ResponseMeta: &schema.ResponseMeta{
			FinishReason: "MAX_TOKENS",
			Usage:        &schema.TokenUsage{PromptTokens: 100, CompletionTokens: 8192, TotalTokens: 8292},
		}}, nil
	})
	out, err := Generate(context.Background(), m, Text("read"), Object(map[string]*jsonschema.Schema{"text": {Type: "string"}}))
	assert.Nil(t, out)
	var exhausted *OutputLimitError
	require.ErrorAs(t, err, &exhausted)
	assert.Equal(t, "MAX_TOKENS", exhausted.FinishReason)
	assert.Equal(t, 8292, exhausted.Usage["TotalTokens"])
}
func TestRetryPolicy(t *testing.T) {
	for _, tc := range []struct {
		name    string
		retries int
		failure string
		want    int
	}{{"transient", 2, "HTTP 503", 3}, {"disabled", 0, "HTTP 429", 1}, {"permanent", 3, "HTTP 400: invalid document 429", 1}} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			m := WithPolicy(modelFunc(func(context.Context, []*schema.Message, ...model.Option) (*schema.Message, error) {
				calls++
				return nil, errors.New(tc.failure)
			}), Policy{MaxRetries: tc.retries, BackoffMaxWait: time.Millisecond})
			_, err := m.Generate(context.Background(), Text("task"))
			require.Error(t, err)
			assert.Equal(t, tc.want, calls)
		})
	}
	t.Run("cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		m := WithPolicy(modelFunc(func(context.Context, []*schema.Message, ...model.Option) (*schema.Message, error) {
			cancel()
			return nil, errors.New("HTTP 429")
		}), Policy{MaxRetries: 3})
		_, err := m.Generate(ctx, Text("task"))
		require.ErrorIs(t, err, context.Canceled)
	})
	t.Run("retry after", func(t *testing.T) {
		assert.Equal(t, 2*time.Second, RetryDelay(errors.New("retry-after-ms:2000"), time.Millisecond))
		assert.Equal(t, time.Second, RetryDelay(errors.New("retry-after-ms:9223372036854775807"), time.Second))
	})
}
func TestEinoOllamaSchemaAndExplicitZero(t *testing.T) {
	var wire map[string]any
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&wire))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"{\"text\":\"page\"}"},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":4}`))
	}))
	defer s.Close()
	zero := 0.0
	think := json.RawMessage(`false`)
	m, err := New(context.Background(), Config{Provider: "ollama", Model: "m", BaseURL: s.URL, Temperature: &zero, OllamaThink: think, OllamaOptions: map[string]any{"num_ctx": 4096, "top_k": 0}})
	require.NoError(t, err)
	out, err := Generate(context.Background(), m, Text("read"), Object(map[string]*jsonschema.Schema{"text": {Type: "string"}}))
	require.NoError(t, err)
	assert.JSONEq(t, `{"text":"page"}`, out.Content)
	assert.Equal(t, false, wire["think"])
	assert.Equal(t, float64(0), wire["options"].(map[string]any)["temperature"])
	assert.Equal(t, float64(0), wire["options"].(map[string]any)["top_k"])
	assert.Equal(t, float64(4096), wire["options"].(map[string]any)["num_ctx"])
	assert.Equal(t, "object", wire["format"].(map[string]any)["type"])
	assert.NotContains(t, wire, "tool_choice")
}
func TestEinoClaudeTemperatureAndTools(t *testing.T) {
	for _, name := range []string{"claude-sonnet-5", "claude-opus-4-7", "claude-haiku-4-5"} {
		t.Run(name, func(t *testing.T) {
			var wire map[string]any
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				require.NoError(t, json.NewDecoder(r.Body).Decode(&wire))
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"id":"r","type":"message","role":"assistant","model":"m","content":[{"type":"tool_use","id":"t","name":"submit_document_result","input":{"text":"page"}}],"stop_reason":"tool_use","usage":{"input_tokens":3,"output_tokens":4}}`))
			}))
			defer s.Close()
			temp := 0.1
			m, err := New(context.Background(), Config{Provider: "anthropic", Model: name, APIKey: "key", BaseURL: s.URL, Temperature: &temp})
			require.NoError(t, err)
			out, err := Generate(context.Background(), m, Text("task"), Object(map[string]*jsonschema.Schema{"text": {Type: "string"}}))
			require.NoError(t, err)
			assert.JSONEq(t, `{"text":"page"}`, out.Content)
			_, sent := wire["temperature"]
			assert.Equal(t, strings.Contains(name, "haiku"), sent)
			assert.Equal(t, "disabled", wire["thinking"].(map[string]any)["type"])
			assert.Equal(t, ResultTool, wire["tool_choice"].(map[string]any)["name"])
		})
	}
}
func TestEinoGeminiThinkingAndTool(t *testing.T) {
	var wire map[string]any
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&wire))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"submit_document_result","args":{"text":"page"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":7}}`))
	}))
	defer s.Close()
	budget := int32(0)
	m, err := New(context.Background(), Config{Provider: "googleai", APIKey: "key", Model: "gemini-2.5-flash", BaseURL: s.URL, ThinkingBudget: &budget})
	require.NoError(t, err)
	pdf := "JVBERi0="
	input := []*schema.Message{{Role: schema.User, UserInputMultiContent: []schema.MessageInputPart{{Type: schema.ChatMessagePartTypeFileURL, File: &schema.MessageInputFile{MessagePartCommon: schema.MessagePartCommon{Base64Data: &pdf, MIMEType: "application/pdf"}}}}}}
	out, err := Generate(context.Background(), m, input, Object(map[string]*jsonschema.Schema{"text": {Type: "string"}}))
	require.NoError(t, err)
	assert.JSONEq(t, `{"text":"page"}`, out.Content)
	assert.Contains(t, wire, "toolConfig")
	assert.Contains(t, wire, "generationConfig")
	part := wire["contents"].([]any)[0].(map[string]any)["parts"].([]any)[0].(map[string]any)["inlineData"].(map[string]any)
	assert.Equal(t, "application/pdf", part["mimeType"])
	assert.Equal(t, pdf, part["data"])
}

func TestOllamaConcurrentOptionsAreIsolated(t *testing.T) {
	var mu sync.Mutex
	temperatures := map[string]float64{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		require.NoError(t, json.NewDecoder(r.Body).Decode(&body))
		prompt := body["messages"].([]any)[0].(map[string]any)["content"].(string)
		mu.Lock()
		temperatures[prompt] = body["options"].(map[string]any)["temperature"].(float64)
		mu.Unlock()
		_, _ = w.Write([]byte(`{"message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop"}`))
	}))
	defer server.Close()
	m, err := New(context.Background(), Config{Provider: "ollama", Model: "m", BaseURL: server.URL})
	require.NoError(t, err)
	var group sync.WaitGroup
	failures := make(chan error, 2)
	for prompt, temp := range map[string]float32{"zero": 0, "other": 0.8} {
		group.Add(1)
		go func(prompt string, temp float32) {
			defer group.Done()
			_, err := m.Generate(context.Background(), Text(prompt), model.WithTemperature(temp))
			failures <- err
		}(prompt, temp)
	}
	group.Wait()
	close(failures)
	for err := range failures {
		require.NoError(t, err)
	}
	assert.Equal(t, float64(0), temperatures["zero"])
	assert.InDelta(t, 0.8, temperatures["other"], 0.0001)
}

func TestGPT5UsesReasoningModelParameters(t *testing.T) {
	var wire map[string]any
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.NoError(t, json.NewDecoder(r.Body).Decode(&wire))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}`))
	}))
	defer s.Close()
	m, err := New(context.Background(), Config{Provider: "openai", APIKey: "key", Model: "gpt-5", BaseURL: s.URL, MaxTokens: 512})
	require.NoError(t, err)
	_, err = Generate(context.Background(), m, Text("task"), nil)
	require.NoError(t, err)
	assert.NotContains(t, wire, "temperature")
	assert.NotContains(t, wire, "max_tokens")
	assert.Equal(t, float64(512), wire["max_completion_tokens"])
}
