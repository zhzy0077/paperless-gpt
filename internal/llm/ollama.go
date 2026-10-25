package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"

	"github.com/cloudwego/eino/components/model"
)

type ollamaSettingsKey struct{}
type ollamaTransport struct{ base http.RoundTripper }

// Preserve explicit zero options omitted by the native adapter's omitempty
// fields. Per-request settings live in the context, so concurrent calls cannot
// change each other's sampling configuration.
func (t ollamaTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	settings, _ := req.Context().Value(ollamaSettingsKey{}).(map[string]any)
	if len(settings) == 0 {
		return t.base.RoundTrip(req)
	}
	raw, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	req.Body.Close()
	var body map[string]any
	if err = json.Unmarshal(raw, &body); err != nil {
		return nil, err
	}
	opts, _ := body["options"].(map[string]any)
	if opts == nil {
		opts = map[string]any{}
	}
	for k, v := range settings {
		if v == nil {
			delete(opts, k)
		} else {
			opts[k] = v
		}
	}
	body["options"] = opts
	raw, err = json.Marshal(body)
	if err != nil {
		return nil, err
	}
	clone := req.Clone(req.Context())
	clone.Body = io.NopCloser(bytes.NewReader(raw))
	clone.ContentLength = int64(len(raw))
	return t.base.RoundTrip(clone)
}
func (p *providerModel) ollamaContext(ctx context.Context, opts []model.Option) context.Context {
	settings := map[string]any{}
	for k, v := range p.config.OllamaOptions {
		settings[k] = v
	}
	if p.config.Temperature != nil {
		settings["temperature"] = *p.config.Temperature
	}
	if p.config.MaxTokens != 0 {
		settings["num_predict"] = p.config.MaxTokens
	}
	explicit := model.GetCommonOptions(&model.Options{}, opts...)
	if explicit.Temperature != nil {
		settings["temperature"] = *explicit.Temperature
	}
	if explicit.TopP != nil {
		settings["top_p"] = *explicit.TopP
	}
	if explicit.Stop != nil {
		settings["stop"] = explicit.Stop
	}
	if explicit.MaxTokens != nil {
		if *explicit.MaxTokens == 0 {
			settings["num_predict"] = nil
		} else {
			settings["num_predict"] = *explicit.MaxTokens
		}
	}
	return context.WithValue(ctx, ollamaSettingsKey{}, settings)
}
