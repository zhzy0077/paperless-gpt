package ocr

import (
	"context"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/stretchr/testify/require"
	"testing"
)

type emptyVision struct{}

func (emptyVision) Generate(context.Context, []*schema.Message, ...model.Option) (*schema.Message, error) {
	return &schema.Message{ResponseMeta: &schema.ResponseMeta{FinishReason: "length"}}, nil
}
func TestProcessImageTokenExhaustionFailsPage(t *testing.T) {
	p := &LLMProvider{llm: emptyVision{}, prompt: "Read"}
	_, err := p.ProcessImage(context.Background(), testJPEG(t), 1)
	require.ErrorContains(t, err, "budget exhausted")
}
