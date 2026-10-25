package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/cloudwego/eino-ext/components/model/claude"
	"github.com/cloudwego/eino-ext/components/model/gemini"
	"github.com/cloudwego/eino-ext/components/model/ollama"
	einoopenai "github.com/cloudwego/eino-ext/components/model/openai"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"google.golang.org/genai"
)

type Config struct {
	Provider, Model, APIKey, BaseURL, APIVersion string
	Azure                                        bool
	HTTPClient                                   *http.Client
	MaxTokens                                    int
	Temperature                                  *float64
	Thinking                                     *bool
	ThinkingBudget                               *int32
	ExtraFields                                  map[string]any
	OllamaOptions                                map[string]any
	OllamaThink                                  json.RawMessage
	OllamaFormat                                 json.RawMessage
	KeepAlive                                    *time.Duration
	Policy                                       Policy
}

type providerModel struct {
	Model
	config       Config
	ollamaConfig *ollama.ChatModelConfig
}

func New(ctx context.Context, c Config) (Model, error) {
	c.Provider = strings.ToLower(c.Provider)
	if c.Provider != "ollama" && c.APIKey == "" {
		return nil, fmt.Errorf("%s API key is not set", c.Provider)
	}
	var m Model
	var err error
	var temp *float32
	if c.Temperature == nil {
		temperature := 0.0

		c.Temperature = &temperature
	}
	if c.Temperature != nil {
		t := float32(*c.Temperature)
		temp = &t
	}
	var max *int
	if c.MaxTokens > 0 {
		v := c.MaxTokens
		max = &v
	}

	p := &providerModel{config: c}
	switch c.Provider {
	case "openai", "mistral":
		if c.Provider == "mistral" && c.BaseURL == "" {
			c.BaseURL = "https://api.mistral.ai/v1"
		}
		extra := map[string]any{}
		for k, v := range c.ExtraFields {
			extra[k] = v
		}
		if c.Thinking != nil {
			extra["thinking"] = map[string]any{"type": map[bool]string{true: "enabled", false: "disabled"}[*c.Thinking]}
		} else if strings.Contains(strings.ToLower(c.Model), "deepseek") {
			extra["thinking"] = map[string]any{"type": "disabled"}
		}
		cfg := &einoopenai.ChatModelConfig{APIKey: c.APIKey, Model: c.Model, BaseURL: c.BaseURL, ByAzure: c.Azure, APIVersion: c.APIVersion, HTTPClient: c.HTTPClient, MaxTokens: max, Temperature: temp, ExtraFields: extra}
		for _, prefix := range []string{"gpt-5", "o1", "o3", "o4"} {
			if strings.HasPrefix(c.Model, prefix) {
				cfg.Temperature = nil
				cfg.MaxTokens = nil
				cfg.MaxCompletionTokens = max
				break
			}
		}
		m, err = einoopenai.NewChatModel(ctx, cfg)

	case "anthropic":
		limit := c.MaxTokens
		if limit <= 0 {
			limit = 8192
		}
		cfg := &claude.Config{APIKey: c.APIKey, Model: c.Model, MaxTokens: limit, Temperature: temp, HTTPClient: c.HTTPClient}
		if strings.HasPrefix(c.Model, "claude-sonnet-5") || strings.HasPrefix(c.Model, "claude-opus-4-7") || strings.HasPrefix(c.Model, "claude-opus-5") {
			cfg.Temperature = nil
		}
		if c.BaseURL != "" {
			cfg.BaseURL = &c.BaseURL
		}
		cfg.ThinkingConfig = &anthropic.ThinkingConfigParamUnion{OfDisabled: &anthropic.ThinkingConfigDisabledParam{}}
		if c.Thinking != nil && *c.Thinking {
			budget := int64(1024)
			if c.ThinkingBudget != nil {
				budget = int64(*c.ThinkingBudget)
			}
			cfg.ThinkingConfig = &anthropic.ThinkingConfigParamUnion{OfEnabled: &anthropic.ThinkingConfigEnabledParam{BudgetTokens: budget}}
			cfg.Temperature = nil
		}
		m, err = claude.NewChatModel(ctx, cfg)
	case "googleai":
		cfg := &genai.ClientConfig{APIKey: c.APIKey, Backend: genai.BackendGeminiAPI, HTTPClient: c.HTTPClient}
		if c.BaseURL != "" {
			cfg.HTTPOptions = genai.HTTPOptions{BaseURL: c.BaseURL}
		}
		var cli *genai.Client
		cli, err = genai.NewClient(ctx, cfg)
		if err != nil {
			return nil, err
		}
		budget := int32(0)
		if c.ThinkingBudget != nil {
			budget = *c.ThinkingBudget
		}
		m, err = gemini.NewChatModel(ctx, &gemini.Config{Client: cli, Model: c.Model, MaxTokens: max, Temperature: temp, ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: &budget}})
	case "ollama":
		if c.BaseURL == "" {
			c.BaseURL = "http://127.0.0.1:11434"
		}
		options := &ollama.Options{}
		raw, _ := json.Marshal(c.OllamaOptions)
		if err = json.Unmarshal(raw, options); err != nil {
			return nil, err
		}
		if c.Temperature != nil {
			options.Temperature = float32(*c.Temperature)
		}
		if c.MaxTokens != 0 {
			options.NumPredict = c.MaxTokens
		}
		var think any = false
		if len(c.OllamaThink) > 0 {
			if err = json.Unmarshal(c.OllamaThink, &think); err != nil {
				return nil, err
			}
		} else if c.Thinking != nil {
			think = *c.Thinking
		}
		client := &http.Client{}
		if c.HTTPClient != nil {
			*client = *c.HTTPClient
		}
		transport := client.Transport
		if transport == nil {
			transport = http.DefaultTransport
		}
		client.Transport = ollamaTransport{base: transport}
		c.HTTPClient = client
		cfg := &ollama.ChatModelConfig{BaseURL: c.BaseURL, Model: c.Model, HTTPClient: c.HTTPClient, Options: options, KeepAlive: c.KeepAlive, Thinking: &ollama.ThinkValue{Value: think}, Format: c.OllamaFormat}
		p.ollamaConfig = cfg
		m, err = ollama.NewChatModel(ctx, cfg)
	default:
		return nil, fmt.Errorf("unsupported LLM provider: %s", c.Provider)
	}
	if err != nil {
		return nil, err
	}
	p.Model = m
	return WithPolicy(p, c.Policy), nil
}

func (p *providerModel) Generate(ctx context.Context, in []*schema.Message, opts ...model.Option) (*schema.Message, error) {
	if p.ollamaConfig != nil {
		ctx = p.ollamaContext(ctx, opts)
	}
	if p.ollamaConfig == nil {
		return p.Model.Generate(ctx, in, opts...)
	}
	// Native Ollama has schema-constrained output but no named tool_choice.
	// Normalize its constrained result into the same result-tool contract.
	common := model.GetCommonOptions(&model.Options{}, opts...)
	if len(common.Tools) == 0 {
		return p.Model.Generate(ctx, in, opts...)
	}
	if len(common.Tools) != 1 {
		return nil, fmt.Errorf("Ollama extraction expects one result tool")
	}
	tool := common.Tools[0]
	spec, err := tool.ParamsOneOf.ToJSONSchema()
	if err != nil {
		return nil, err
	}
	cfg := *p.ollamaConfig
	cfg.Format, err = json.Marshal(spec)
	if err != nil {
		return nil, err
	}
	m, err := ollama.NewChatModel(ctx, &cfg)
	if err != nil {
		return nil, err
	}
	opts = append(opts, model.WithTools(nil), model.WithToolChoice(schema.ToolChoiceForbidden))
	out, err := m.Generate(ctx, in, opts...)
	if err != nil || out == nil {
		return out, err
	}
	clone := *out
	clone.ToolCalls = []schema.ToolCall{{Type: "function", Function: schema.FunctionCall{Name: tool.Name, Arguments: out.Content}}}
	return &clone, nil
}
