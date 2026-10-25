package ocr

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/cloudwego/eino/schema"
	"github.com/eino-contrib/jsonschema"
	"github.com/gabriel-vasile/mimetype"
	"paperless-gpt/internal/llm"
	"paperless-gpt/sanitize"
)

type LLMProvider struct {
	provider, model, prompt string
	llm                     llm.Model
}

func (p *LLMProvider) WithPrompt(prompt string) *LLMProvider {
	clone := *p
	clone.prompt = prompt
	return &clone
}
func (p *LLMProvider) GetPrompt() string { return p.prompt }
func newLLMProvider(config Config) (*LLMProvider, error) {
	c := llm.Config{Provider: config.VisionLLMProvider, Model: config.VisionLLMModel, MaxTokens: config.VisionLLMMaxTokens, Temperature: config.VisionLLMTemperature, ThinkingBudget: config.GoogleAIThinkingBudget}
	if c.MaxTokens <= 0 {
		c.MaxTokens = 8192
	}
	c.Policy = llm.Policy{RequestsPerMinute: 120, MaxRetries: 8, BackoffMaxWait: 90 * time.Second}
	if v, e := strconv.Atoi(os.Getenv("VISION_LLM_MAX_RETRIES")); e == nil && v >= 0 {
		c.Policy.MaxRetries = v
	}
	if v, e := time.ParseDuration(os.Getenv("VISION_LLM_BACKOFF_MAX_WAIT")); e == nil && v > 0 {
		c.Policy.BackoffMaxWait = v
	}
	if v, e := strconv.ParseFloat(os.Getenv("VISION_LLM_REQUESTS_PER_MINUTE"), 64); e == nil {
		c.Policy.RequestsPerMinute = v
	}
	switch strings.ToLower(c.Provider) {
	case "openai":
		c.APIKey = os.Getenv("OPENAI_API_KEY")
		c.BaseURL = os.Getenv("OPENAI_BASE_URL")
		c.HTTPClient = OpenAIHTTPClient()
		c.Azure = strings.EqualFold(os.Getenv("OPENAI_API_TYPE"), "azure")
		c.APIVersion = os.Getenv("OPENAI_API_VERSION")
	case "mistral":
		c.APIKey = os.Getenv("MISTRAL_API_KEY")
	case "anthropic":
		c.APIKey = os.Getenv("ANTHROPIC_API_KEY")
	case "googleai":
		c.APIKey = config.GoogleAIAPIKey
	case "ollama":
		c.BaseURL = os.Getenv("OLLAMA_HOST")
		c.HTTPClient = OllamaHTTPClient()
		if c.HTTPClient == nil {
			c.HTTPClient = defaultOllamaClient()
		} else {
			c.HTTPClient.Timeout = defaultOllamaClient().Timeout
		}
		c.OllamaOptions = map[string]any{}
		if config.OllamaContextLength > 0 {
			c.OllamaOptions["num_ctx"] = config.OllamaContextLength
		}
		if config.OllamaOcrTopK != nil {
			c.OllamaOptions["top_k"] = *config.OllamaOcrTopK
		}
	}
	m, err := llm.New(context.Background(), c)
	if err != nil {
		return nil, err
	}
	return &LLMProvider{provider: c.Provider, model: c.Model, prompt: config.VisionLLMPrompt, llm: m}, nil
}
func (p *LLMProvider) ProcessImage(ctx context.Context, data []byte, pageNumber int) (*OCRResult, error) {
	mediaType := mimetype.Detect(data).String()
	if mediaType != "application/pdf" {
		if _, _, err := image.Decode(bytes.NewReader(data)); err != nil {
			return nil, fmt.Errorf("error decoding image: %w", err)
		}
	}
	b64 := base64.StdEncoding.EncodeToString(data)
	part := schema.MessageInputPart{Type: schema.ChatMessagePartTypeImageURL, Image: &schema.MessageInputImage{MessagePartCommon: schema.MessagePartCommon{Base64Data: &b64, MIMEType: mediaType}}}
	if mediaType == "application/pdf" {
		part = schema.MessageInputPart{Type: schema.ChatMessagePartTypeFileURL, File: &schema.MessageInputFile{MessagePartCommon: schema.MessagePartCommon{Base64Data: &b64, MIMEType: mediaType}}}
	}
	spec := llm.Object(map[string]*jsonschema.Schema{"text": {Type: "string", Description: "Complete transcription of the document page, preserving the text, row associations and formatting. Include only document content."}})
	out, err := llm.Generate(ctx, p.llm, []*schema.Message{{Role: schema.User, UserInputMultiContent: []schema.MessageInputPart{part, {Type: schema.ChatMessagePartTypeText, Text: sanitize.Sanitize(p.prompt)}}}}, spec)
	if err != nil {
		return nil, fmt.Errorf("OCR page %d: %w", pageNumber, err)
	}
	var value struct {
		Text string `json:"text"`
	}
	if err = json.Unmarshal([]byte(out.Content), &value); err != nil {
		return nil, err
	}
	return &OCRResult{Text: value.Text, Metadata: map[string]string{"provider": p.provider, "model": p.model}, GenerationInfo: llm.Usage(out)}, nil
}

func defaultOllamaClient() *http.Client {
	timeout := 300 * time.Second
	if raw := os.Getenv("OLLAMA_TIMEOUT_SECONDS"); raw != "" {
		if seconds, err := strconv.Atoi(raw); err == nil {
			timeout = time.Duration(seconds) * time.Second
			if seconds <= 0 {
				timeout = 0
			}
		}
	}
	return &http.Client{Timeout: timeout}
}
