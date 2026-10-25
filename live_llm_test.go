package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"paperless-gpt/internal/llm"
	"paperless-gpt/ocr"
)

// Opt-in live acceptance test. The ordinary suite has no API-key requirement.
func TestLiveEinoMigration(t *testing.T) {
	imagePath := os.Getenv("EINO_EVAL_IMAGE")
	if imagePath == "" {
		t.Skip("set EINO_EVAL_IMAGE and OPENAI_API_KEY to run live acceptance")
	}
	image, err := os.ReadFile(imagePath)
	require.NoError(t, err)
	goldRaw, err := os.ReadFile(os.Getenv("EINO_EVAL_GOLD"))
	require.NoError(t, err)
	var gold struct {
		Facts map[string]string `json:"facts"`
		Rows  [][]any           `json:"rows"`
	}
	require.NoError(t, json.Unmarshal(goldRaw, &gold))
	modelName := os.Getenv("EINO_EVAL_MODEL")
	if modelName == "" {
		modelName = "deepseek-v4.1-flash"
	}
	prompt, err := os.ReadFile("default_prompts/ocr_prompt.tmpl")
	require.NoError(t, err)
	prompt = []byte(strings.Split(string(prompt), "{{- if .Content}}")[0])
	metadata, err := llm.New(context.Background(), llm.Config{Provider: "openai", Model: modelName, APIKey: os.Getenv("OPENAI_API_KEY"), BaseURL: os.Getenv("OPENAI_BASE_URL"), MaxTokens: 8192})
	require.NoError(t, err)
	provider, err := ocr.NewProvider(ocr.Config{Provider: "llm", VisionLLMProvider: "openai", VisionLLMModel: modelName, VisionLLMPrompt: string(prompt), VisionLLMMaxTokens: 8192})
	require.NoError(t, err)
	setupMetadataTest(t)
	fields := []CustomField{{ID: 1, Name: "账单号", DataType: "string"}, {ID: 2, Name: "账单合计", DataType: "float"}, {ID: 3, Name: "优惠金额", DataType: "float"}, {ID: 4, Name: "支付金额", DataType: "float"}, {ID: 5, Name: "台号", DataType: "integer"}, {ID: 6, Name: "开台时间", DataType: "string"}, {ID: 7, Name: "打印时间", DataType: "string"}, {ID: 8, Name: "店铺地址", DataType: "string"}, {ID: 9, Name: "结账人", DataType: "string"}, {ID: 10, Name: "订单渠道", DataType: "string"}}
	settings.CustomFieldsEnable = true
	for _, f := range fields {
		settings.CustomFieldsSelectedIDs = append(settings.CustomFieldsSelectedIDs, f.ID)
	}
	app := &App{LLM: metadata, Client: &mockPaperlessClient{CustomFields: fields}}
	req := GenerateSuggestionsRequest{GenerateTitles: true, GenerateTags: true, GenerateCorrespondents: true, GenerateDocumentTypes: true, GenerateCreatedDate: true, GenerateCustomFields: true}
	candidates := suggestionGenerationContext{availableTagNames: []string{"餐饮", "消费", "医疗", "合同", "工资"}, availableCorrespondentNames: []string{gold.Facts["merchant"], "国家电网", "中国移动"}, availableDocumentTypeNames: []string{"收据", "合同", "信函", "病历"}}
	integer := regexp.MustCompile(`\d+`)
	for rep := 1; rep <= 3; rep++ {
		t.Run(strconv.Itoa(rep), func(t *testing.T) {
			result, err := provider.ProcessImage(context.Background(), image, 1)
			require.NoError(t, err)
			rows := 0
			for _, row := range gold.Rows {
				name := row[0].(string)
				want := int(row[1].(float64))
				unit := row[2].(string)
				for _, line := range strings.Split(result.Text, "\n") {
					i := strings.Index(line, name)
					if i < 0 {
						continue
					}
					tail := line[i+len(name):]
					n := integer.FindString(tail)
					got, _ := strconv.Atoi(n)
					if n != "" && got == want && strings.Contains(tail, unit) {
						rows++
						break
					}
				}
			}
			assert.Equal(t, len(gold.Rows), rows, "item/quantity/unit rows")
			suggestion, err := app.generateSingleDocumentSuggestion(context.Background(), req, Document{ID: 1, Title: "OCR.pdf", Content: result.Text}, candidates, logrus.NewEntry(log))
			require.NoError(t, err)
			assert.Contains(t, suggestion.SuggestedTitle, gold.Facts["merchant"])
			assert.Equal(t, gold.Facts["merchant"], suggestion.SuggestedCorrespondent)
			assert.Equal(t, "收据", suggestion.SuggestedDocumentType)
			assert.Equal(t, strings.Fields(gold.Facts["opened"])[0], suggestion.SuggestedCreatedDate)
			assert.ElementsMatch(t, []string{"餐饮", "消费"}, suggestion.SuggestedTags)
			require.Len(t, suggestion.SuggestedCustomFields, 10)
			number := func(key string) float64 {
				v, e := strconv.ParseFloat(gold.Facts[key], 64)
				require.NoError(t, e)
				return v
			}
			expected := map[int]any{1: gold.Facts["bill"], 2: number("total"), 3: number("discount"), 4: number("paid"), 5: number("table"), 6: gold.Facts["opened"], 7: gold.Facts["printed"], 8: gold.Facts["address"], 9: gold.Facts["cashier"], 10: gold.Facts["channel"]}
			for _, f := range suggestion.SuggestedCustomFields {
				assert.Equal(t, expected[f.ID], f.Value, "field %s", f.Name)
			}
			if dir := os.Getenv("EINO_EVAL_OUT"); dir != "" {
				b, _ := json.MarshalIndent(map[string]any{"ocr": result, "suggestion": suggestion, "correct_rows": rows}, "", "  ")
				require.NoError(t, os.WriteFile(filepath.Join(dir, "rewritten-"+strconv.Itoa(rep)+".json"), b, 0600))
			}
			t.Logf("OCR %d/%d rows; all ten custom fields checked; OCR usage=%v", rows, len(gold.Rows), result.GenerationInfo)
		})
	}
}
