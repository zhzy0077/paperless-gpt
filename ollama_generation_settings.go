package main

import (
	"encoding/json"
	"fmt"
	"math"
	"time"

	ollamaapi "github.com/cloudwego/eino-ext/components/model/ollama"
)

// OllamaMetadataConfig configures metadata generation requests sent to native
// Ollama.
type OllamaMetadataConfig struct {
	Defaults OllamaGenerationSettings `json:"defaults,omitempty"`
}

// OllamaGenerationSettings contains optional Ollama chat generation settings.
// Pointers intentionally distinguish an omitted setting from an explicit zero
// value. Think and Format are raw JSON because Ollama accepts more than one
// JSON type for each of them.
type OllamaGenerationSettings struct {
	Temperature      *float64        `json:"temperature,omitempty"`
	MaxTokens        *int            `json:"max_tokens,omitempty"`
	ContextLength    *int            `json:"num_ctx,omitempty"`
	KeepAlive        *string         `json:"keep_alive,omitempty"`
	Think            json.RawMessage `json:"think,omitempty"`
	Format           json.RawMessage `json:"format,omitempty"`
	TopK             *int            `json:"top_k,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	MinP             *float64        `json:"min_p,omitempty"`
	Seed             *int            `json:"seed,omitempty"`
	RepeatPenalty    *float64        `json:"repeat_penalty,omitempty"`
	RepeatLastN      *int            `json:"repeat_last_n,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	Stop             *[]string       `json:"stop,omitempty"`
}

// Validate checks syntax and request-level invariants. Ollama remains
// responsible for validating a JSON schema's semantics.
func (s OllamaGenerationSettings) Validate() error {
	for _, value := range []*float64{s.Temperature, s.TopP, s.MinP, s.RepeatPenalty, s.FrequencyPenalty, s.PresencePenalty} {
		if value != nil && (math.IsNaN(*value) || math.IsInf(*value, 0)) {
			return fmt.Errorf("Ollama generation setting must be a finite number")
		}
	}
	if s.Temperature != nil && *s.Temperature < 0 {
		return fmt.Errorf("Ollama temperature must be non-negative")
	}
	if s.MaxTokens != nil && (*s.MaxTokens == 0 || *s.MaxTokens < -1) {
		return fmt.Errorf("Ollama max_tokens must be positive or -1")
	}
	if s.ContextLength != nil && *s.ContextLength <= 0 {
		return fmt.Errorf("Ollama num_ctx must be positive")
	}
	if s.TopK != nil && *s.TopK < 0 {
		return fmt.Errorf("Ollama top_k must be non-negative")
	}
	if s.TopP != nil && (*s.TopP < 0 || *s.TopP > 1) {
		return fmt.Errorf("Ollama top_p must be between 0 and 1")
	}
	if s.MinP != nil && (*s.MinP < 0 || *s.MinP > 1) {
		return fmt.Errorf("Ollama min_p must be between 0 and 1")
	}
	if s.RepeatPenalty != nil && *s.RepeatPenalty < 0 {
		return fmt.Errorf("Ollama repeat_penalty must be non-negative")
	}
	if s.RepeatLastN != nil && *s.RepeatLastN < -1 {
		return fmt.Errorf("Ollama repeat_last_n must be -1 or greater")
	}
	if s.KeepAlive != nil {
		if _, err := parseOllamaKeepAlive(*s.KeepAlive); err != nil {
			return err
		}
	}
	if len(s.Think) > 0 {
		if _, err := ollamaThinkValue(s.Think); err != nil {
			return err
		}
	}
	if len(s.Format) > 0 {
		if err := validateOllamaFormat(s.Format); err != nil {
			return err
		}
	}
	return nil
}

func parseOllamaKeepAlive(value string) (*time.Duration, error) {
	if value == "0" {
		value := time.Duration(0)
		return &value, nil
	}
	if value == "-1" {
		value := time.Duration(-1)
		return &value, nil
	}
	duration, err := time.ParseDuration(value)
	if err != nil || duration < 0 {
		return nil, fmt.Errorf("invalid Ollama keep_alive %q", value)
	}
	return &duration, nil
}

func ollamaThinkValue(raw json.RawMessage) (*ollamaapi.ThinkValue, error) {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, fmt.Errorf("invalid Ollama think: %w", err)
	}
	switch typed := value.(type) {
	case bool:
		return &ollamaapi.ThinkValue{Value: typed}, nil
	case string:
		if typed == "low" || typed == "medium" || typed == "high" {
			return &ollamaapi.ThinkValue{Value: typed}, nil
		}
	}
	return nil, fmt.Errorf("invalid Ollama think (expected true, false, low, medium, or high)")
}

func validateOllamaFormat(raw json.RawMessage) error {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return fmt.Errorf("invalid Ollama format: %w", err)
	}
	if value == "json" {
		return nil
	}
	if schema, ok := value.(map[string]any); ok && len(schema) > 0 {
		return nil
	}
	return fmt.Errorf("invalid Ollama format (expected \"json\" or a non-empty JSON object)")
}
