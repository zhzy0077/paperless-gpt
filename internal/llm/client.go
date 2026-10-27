// Package llm provides the application's model boundary using Eino adapters.
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/eino-contrib/jsonschema"
)

// Model is the synchronous portion of Eino's ChatModel used by the application.
type Model interface {
	Generate(context.Context, []*schema.Message, ...model.Option) (*schema.Message, error)
}

const ResultTool = "submit_document_result"

// OutputLimitError preserves usage when an incomplete generation is rejected.
type OutputLimitError struct {
	FinishReason string
	Usage        map[string]any
}

func (e *OutputLimitError) Error() string {
	return fmt.Sprintf("LLM output budget exhausted (usage: %+v)", e.Usage)
}

// Generate returns either free-form text or validated arguments of the result tool.
// The schema is sent exclusively through the tool definition.
func Generate(ctx context.Context, client Model, messages []*schema.Message, spec *jsonschema.Schema, opts ...model.Option) (*schema.Message, error) {
	if client == nil {
		return nil, errors.New("LLM is not configured")
	}
	if spec != nil {
		opts = append(opts, model.WithTools([]*schema.ToolInfo{{Name: ResultTool, Desc: "Submit the complete result of the requested document processing task.", ParamsOneOf: schema.NewParamsOneOfByJSONSchema(spec)}}), model.WithToolChoice(schema.ToolChoiceForced, ResultTool))
	}
	out, err := client.Generate(ctx, messages, opts...)
	if err != nil {
		return nil, err
	}
	if out == nil {
		return nil, errors.New("LLM returned no response")
	}
	if out.ResponseMeta != nil && (strings.EqualFold(out.ResponseMeta.FinishReason, "length") || strings.EqualFold(out.ResponseMeta.FinishReason, "max_tokens")) {
		return nil, &OutputLimitError{FinishReason: out.ResponseMeta.FinishReason, Usage: Usage(out)}
	}
	if spec != nil {
		if len(out.ToolCalls) != 1 || out.ToolCalls[0].Function.Name != ResultTool {
			return nil, errors.New("LLM did not return the requested result tool")
		}
		text := out.ToolCalls[0].Function.Arguments
		var value any
		if err := json.Unmarshal([]byte(text), &value); err != nil {
			return nil, fmt.Errorf("invalid tool arguments: %w", err)
		}
		// Validate locally even when an endpoint accepts but ignores strict constraints.
		if err := Validate(value, spec); err != nil {
			return nil, fmt.Errorf("invalid tool result: %w", err)
		}
		clone := *out
		clone.Content = text
		out = &clone
	}
	return out, nil
}

// Object builds a closed object schema, requiring exactly the requested fields.
func Object(properties map[string]*jsonschema.Schema) *jsonschema.Schema {
	raw := map[string]any{"type": "object", "properties": properties, "additionalProperties": false}
	required := make([]string, 0, len(properties))
	for k := range properties {
		required = append(required, k)
	}
	sort.Strings(required)
	raw["required"] = required
	b, _ := json.Marshal(raw)
	s := new(jsonschema.Schema)
	_ = json.Unmarshal(b, s)
	return s
}

// Validate checks the schema subset used by extraction, without coercing values.
func Validate(value any, spec *jsonschema.Schema) error {
	b, err := json.Marshal(spec)
	if err != nil {
		return err
	}
	var s map[string]any
	if err = json.Unmarshal(b, &s); err != nil {
		return err
	}
	return validate(value, s, "result")
}
func validate(v any, s map[string]any, path string) error {
	typ, _ := s["type"].(string)
	switch typ {
	case "object":
		obj, ok := v.(map[string]any)
		if !ok {
			return fmt.Errorf("%s must be an object", path)
		}
		props, _ := s["properties"].(map[string]any)
		required, _ := s["required"].([]any)
		for _, k := range required {
			if _, ok := obj[k.(string)]; !ok {
				return fmt.Errorf("%s.%s is missing", path, k)
			}
		}
		for k, val := range obj {
			p, known := props[k]
			if !known {
				if s["additionalProperties"] == false {
					return fmt.Errorf("%s.%s is not requested", path, k)
				}
				continue
			}
			if err := validate(val, p.(map[string]any), path+"."+k); err != nil {
				return err
			}
		}
	case "array":
		arr, ok := v.([]any)
		if !ok {
			return fmt.Errorf("%s must be an array", path)
		}
		item, _ := s["items"].(map[string]any)
		for i, x := range arr {
			if err := validate(x, item, fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
	case "string":
		if _, ok := v.(string); !ok {
			return fmt.Errorf("%s must be a string", path)
		}
	case "number", "integer":
		n, ok := v.(float64)
		if !ok || (typ == "integer" && math.Trunc(n) != n) {
			return fmt.Errorf("%s must be %s", path, typ)
		}
	case "boolean":
		if _, ok := v.(bool); !ok {
			return fmt.Errorf("%s must be boolean", path)
		}
	}
	if values, ok := s["enum"].([]any); ok {
		for _, x := range values {
			if fmt.Sprint(x) == fmt.Sprint(v) {
				return nil
			}
		}
		return fmt.Errorf("%s has an unsupported value", path)
	}
	if choices, ok := s["anyOf"].([]any); ok {
		for _, x := range choices {
			if validate(v, x.(map[string]any), path) == nil {
				return nil
			}
		}
		return fmt.Errorf("%s has an unsupported type", path)
	}
	return nil
}

func Text(prompt string) []*schema.Message {
	return []*schema.Message{{Role: schema.User, Content: prompt}}
}
func Usage(out *schema.Message) map[string]any {
	info := map[string]any{}
	if out != nil && out.ResponseMeta != nil && out.ResponseMeta.Usage != nil {
		u := out.ResponseMeta.Usage
		info["PromptTokens"] = u.PromptTokens
		info["CompletionTokens"] = u.CompletionTokens
		info["TotalTokens"] = u.TotalTokens
		info["ReasoningTokens"] = u.CompletionTokensDetails.ReasoningTokens
	}
	return info
}

var statusPattern = regexp.MustCompile(`(?i)(?:status(?:[ _-]*code)?[=: ]*|http\s+|error\s+|:\s+)([45][0-9]{2})\b`)

func IsTransient(err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	match := statusPattern.FindStringSubmatch(err.Error())
	if len(match) < 2 {
		return false
	}
	return match[1] == "429" || match[1][0] == '5'
}
