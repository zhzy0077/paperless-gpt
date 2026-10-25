package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/gorm"
	"os"
	"paperless-gpt/internal/llm"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type mockLLM struct {
	lastPrompt string
	Response   string
	Error      error
	calls      int
	tool       *schema.ToolInfo
}

func (m *mockLLM) Generate(ctx context.Context, in []*schema.Message, opts ...model.Option) (*schema.Message, error) {
	m.calls++
	if len(in) > 0 {
		m.lastPrompt = in[0].Content
	}
	if m.Error != nil {
		return nil, m.Error
	}
	options := model.GetCommonOptions(&model.Options{}, opts...)
	if len(options.Tools) == 0 {
		return &schema.Message{Content: m.Response}, nil
	}
	m.tool = options.Tools[0]
	response := m.Response
	if !json.Valid([]byte(response)) {
		spec, _ := m.tool.ParamsOneOf.ToJSONSchema()
		var raw map[string]any
		b, _ := json.Marshal(spec)
		_ = json.Unmarshal(b, &raw)
		values := map[string]any{}
		for k := range raw["properties"].(map[string]any) {
			switch k {
			case "title":
				if response == "" {
					values[k] = "test response"
				} else {
					values[k] = response
				}
			case "tags":
				values[k] = []string{}
			case "custom_fields":
				values[k] = map[string]any{}
			default:
				values[k] = ""
			}
		}
		b, _ = json.Marshal(values)
		response = string(b)
	}
	return &schema.Message{ToolCalls: []schema.ToolCall{{Type: "function", Function: schema.FunctionCall{Name: llm.ResultTool, Arguments: response}}}}, nil
}

func setupMetadataTest(t *testing.T) {
	t.Helper()
	oldLimit, oldSettings := tokenLimit, settings
	oldTitle, oldTag, oldCorr, oldType, oldDate, oldCustom := titleTemplate, tagTemplate, correspondentTemplate, documentTypeTemplate, createdDateTemplate, customFieldTemplate
	t.Cleanup(func() {
		tokenLimit = oldLimit
		settings = oldSettings
		titleTemplate = oldTitle
		tagTemplate = oldTag
		correspondentTemplate = oldCorr
		documentTypeTemplate = oldType
		createdDateTemplate = oldDate
		customFieldTemplate = oldCustom
	})
	tokenLimit = 0
	usePromptsSandbox(t, nil)
	require.NoError(t, loadTemplates())
}
func TestPrepareSuggestionGenerationContextFetchesOnlyRequestedMetadata(t *testing.T) {
	app := &App{
		Client: &mockPaperlessClient{
			TagsError:           fmt.Errorf("tags should not be fetched"),
			CorrespondentsError: fmt.Errorf("correspondents should not be fetched"),
			DocumentTypesError:  fmt.Errorf("document types should not be fetched"),
		},
	}

	_, err := app.prepareSuggestionGenerationContext(context.Background(), GenerateSuggestionsRequest{
		GenerateTitles:      true,
		GenerateCreatedDate: true,
	})
	require.NoError(t, err)

	client, ok := app.Client.(*mockPaperlessClient)
	require.True(t, ok, "Client should be *mockPaperlessClient")
	assert.Zero(t, client.GetAllTagsCalls)
	assert.Zero(t, client.GetAllCorrespondentsCalls)
	assert.Zero(t, client.GetAllDocumentTypesCalls)

	app = &App{Client: &mockPaperlessClient{}}
	contextData, err := app.prepareSuggestionGenerationContext(context.Background(), GenerateSuggestionsRequest{
		GenerateTags:           true,
		GenerateCorrespondents: true,
		GenerateDocumentTypes:  true,
	})
	require.NoError(t, err)

	client, ok = app.Client.(*mockPaperlessClient)
	require.True(t, ok, "Client should be *mockPaperlessClient")
	assert.Equal(t, 1, client.GetAllTagsCalls)
	assert.Equal(t, 1, client.GetAllCorrespondentsCalls)
	assert.Equal(t, 1, client.GetAllDocumentTypesCalls)
	assert.Equal(t, []string{"invoice"}, contextData.availableTagNames)
	assert.Equal(t, []string{"Vendor"}, contextData.availableCorrespondentNames)
	assert.Equal(t, []string{"Invoice"}, contextData.availableDocumentTypeNames)
}

// mockPaperlessClient is a mock implementation of the ClientInterface for testing.
type mockPaperlessClient struct {
	CustomFields              []CustomField
	Error                     error
	TagsError                 error
	CorrespondentsError       error
	DocumentTypesError        error
	GetAllTagsCalls           int
	GetAllCorrespondentsCalls int
	GetAllDocumentTypesCalls  int
	// ReferenceMatches maps a reference to the document ids
	// FindDocumentIDsByReference returns for it.
	ReferenceMatches map[string][]int
	ReferenceError   error
	ReferenceLookups []string
}

func (m *mockPaperlessClient) GetCustomFields(ctx context.Context) ([]CustomField, error) {
	if m.Error != nil {
		return nil, m.Error
	}
	return m.CustomFields, nil
}

// Implement other methods of the interface with empty bodies as they are not needed for this test.
func (m *mockPaperlessClient) GetDocumentsByTag(ctx context.Context, tag string, pageSize int) ([]Document, error) {
	return nil, nil
}
func (m *mockPaperlessClient) GetDocumentCountByTag(ctx context.Context, tag string) (int, error) {
	return 0, nil
}
func (m *mockPaperlessClient) UpdateDocuments(ctx context.Context, documents []DocumentSuggestion, db *gorm.DB, isUndo bool) error {
	return nil
}
func (m *mockPaperlessClient) GetDocument(ctx context.Context, documentID int) (Document, error) {
	return Document{}, nil
}
func (m *mockPaperlessClient) GetDocumentThumbnail(ctx context.Context, documentID int) ([]byte, string, error) {
	return nil, "", nil
}
func (m *mockPaperlessClient) SearchDocuments(ctx context.Context, query string, pageSize int) ([]Document, error) {
	return nil, nil
}
func (m *mockPaperlessClient) FindDocumentIDsByReference(ctx context.Context, reference string, excludeID, maxMatches int) ([]int, bool, error) {
	m.ReferenceLookups = append(m.ReferenceLookups, reference)
	if m.ReferenceError != nil {
		return nil, false, m.ReferenceError
	}
	var ids []int
	for _, id := range m.ReferenceMatches[reference] {
		if id != excludeID {
			ids = append(ids, id)
		}
	}
	if len(ids) > maxMatches {
		return nil, true, nil
	}
	return ids, false, nil
}
func (m *mockPaperlessClient) GetDocumentPageImage(ctx context.Context, documentID int, pageIndex int) ([]byte, error) {
	return nil, nil
}
func (m *mockPaperlessClient) GetAllTags(ctx context.Context) (map[string]int, error) {
	m.GetAllTagsCalls++
	if m.TagsError != nil {
		return nil, m.TagsError
	}
	return map[string]int{"invoice": 1, manualTag: 2}, nil
}
func (m *mockPaperlessClient) GetAllCorrespondents(ctx context.Context) (map[string]int, error) {
	m.GetAllCorrespondentsCalls++
	if m.CorrespondentsError != nil {
		return nil, m.CorrespondentsError
	}
	return map[string]int{"Vendor": 1}, nil
}
func (m *mockPaperlessClient) GetAllDocumentTypes(ctx context.Context) ([]DocumentType, error) {
	m.GetAllDocumentTypesCalls++
	if m.DocumentTypesError != nil {
		return nil, m.DocumentTypesError
	}
	return []DocumentType{{ID: 1, Name: "Invoice"}}, nil
}
func (m *mockPaperlessClient) CreateTag(ctx context.Context, tagName string) (int, error) {
	return 0, nil
}
func (m *mockPaperlessClient) DownloadDocumentAsImages(ctx context.Context, documentID int, pageLimit int) ([]string, int, error) {
	return nil, 0, nil
}
func (m *mockPaperlessClient) DownloadDocumentAsPDF(ctx context.Context, documentID int, limitPages int, split bool) ([]string, []byte, int, error) {
	return nil, nil, 0, nil
}
func (m *mockPaperlessClient) UploadDocument(ctx context.Context, data []byte, filename string, metadata map[string]interface{}) (string, error) {
	return "", nil
}
func (m *mockPaperlessClient) UploadDocumentVersion(ctx context.Context, documentID int, data []byte, filename string, versionLabel string) (string, error) {
	return "", nil
}
func (m *mockPaperlessClient) GetTaskStatus(ctx context.Context, taskID string) (map[string]interface{}, error) {
	return nil, nil
}
func (m *mockPaperlessClient) DeleteDocument(ctx context.Context, documentID int) error { return nil }

func TestHandoverTags(t *testing.T) {
	previousManualTag, previousAutoTag := manualTag, autoTag
	manualTag, autoTag = "paperless-gpt", "paperless-gpt-auto"
	t.Cleanup(func() { manualTag, autoTag = previousManualTag, previousAutoTag })

	invoices := &WorkflowConfig{ID: "wf1", TriggerTag: "invoices", CompletionTag: "invoices-done"}
	noCompletion := &WorkflowConfig{ID: "wf2", TriggerTag: "contracts"}

	tests := []struct {
		name       string
		complete   string
		auto       bool
		trigger    string
		workflow   *WorkflowConfig
		wantRemove []string
		wantAdd    []string
	}{
		{"manual review adds no completion tag", "done", false, "", nil,
			[]string{"paperless-gpt", "paperless-gpt-auto"}, nil},
		{"default path adds AUTO_TAG_COMPLETE", "done", true, "paperless-gpt-auto", nil,
			[]string{"paperless-gpt", "paperless-gpt-auto"}, []string{"done"}},
		{"workflow completion tag replaces AUTO_TAG_COMPLETE", "done", true, "invoices", invoices,
			[]string{"paperless-gpt", "paperless-gpt-auto", "invoices"}, []string{"invoices-done"}},
		{"workflow without completion tag falls back to AUTO_TAG_COMPLETE", "done", true, "contracts", noCompletion,
			[]string{"paperless-gpt", "paperless-gpt-auto", "contracts"}, []string{"done"}},
		{"no completion tag configured anywhere", "", true, "contracts", noCompletion,
			[]string{"paperless-gpt", "paperless-gpt-auto", "contracts"}, nil},
		{"trigger equal to AUTO_TAG is not listed twice", "", true, "Paperless-GPT-Auto", nil,
			[]string{"paperless-gpt", "paperless-gpt-auto"}, nil},
		{"the polled tag comes off, whatever the workflow says", "", true, "routed-tag", invoices,
			[]string{"paperless-gpt", "paperless-gpt-auto", "routed-tag"}, []string{"invoices-done"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			remove, add := handoverTags(tt.complete, tt.auto, tt.trigger, tt.workflow)
			assert.Equal(t, tt.wantRemove, remove)
			assert.Equal(t, tt.wantAdd, add)
		})
	}
}

// A selected custom field may itself be called "field". A map reply for it
// must not be mistaken for a single {"field": ..., "value": ...} object.
func usePromptsSandbox(t *testing.T, prompts map[string]string) {
	t.Helper()
	defaults, err := filepath.Abs("default_prompts")
	require.NoError(t, err)
	dir := t.TempDir()
	require.NoError(t, os.CopyFS(filepath.Join(dir, "default_prompts"), os.DirFS(defaults)))
	t.Chdir(dir)
	require.NoError(t, os.MkdirAll("prompts", 0o755))
	for name, content := range prompts {
		require.NoError(t, os.WriteFile(filepath.Join("prompts", name), []byte(content), 0o644))
	}
}

// A prompt in prompts/ that is still an earlier default was never customised
// and gets the current default; a customised prompt is left alone.
func TestLoadTemplatesUpdatesOutdatedDefaults(t *testing.T) {
	const outdated = "an earlier default custom field prompt {{ .CustomFieldsXML }}"
	sum := sha256.Sum256([]byte(outdated))
	previous := previousDefaultPrompts
	previousDefaultPrompts = map[string][]string{"custom_field_prompt.tmpl": {hex.EncodeToString(sum[:])}}
	t.Cleanup(func() { previousDefaultPrompts = previous })

	usePromptsSandbox(t, map[string]string{
		"custom_field_prompt.tmpl": outdated,
		"title_prompt.tmpl":        "my own title prompt {{ .Content }}",
	})
	require.NoError(t, loadTemplates())

	current, err := os.ReadFile("default_prompts/custom_field_prompt.tmpl")
	require.NoError(t, err)
	got, err := os.ReadFile("prompts/custom_field_prompt.tmpl")
	require.NoError(t, err)
	assert.Equal(t, string(current), string(got), "the outdated default is replaced")

	got, err = os.ReadFile("prompts/title_prompt.tmpl")
	require.NoError(t, err)
	assert.Equal(t, "my own title prompt {{ .Content }}", string(got), "a customised prompt is kept")
}

func TestPreviousDefaultPromptsAreForExistingPrompts(t *testing.T) {
	for name, sums := range previousDefaultPrompts {
		current, err := os.ReadFile(filepath.Join("default_prompts", name))
		require.NoError(t, err, "history entry for a prompt that no longer exists: %s", name)
		sum := sha256.Sum256(current)
		assert.NotContains(t, sums, hex.EncodeToString(sum[:]), "%s: the current default must not be listed as outdated", name)
	}
}

func TestMetadataCombinedRequest(t *testing.T) {
	setupMetadataTest(t)
	settings.CustomFieldsSelectedIDs = []int{1, 2}
	settings.CustomFieldsEnable = true
	m := &mockLLM{Response: `{"title":"ACME invoice","tags":["INVOICE","paperless-gpt-auto","new"],"correspondent":"Vendor","document_type":"invoice","created_date":"2023.01.01","custom_fields":{"1":"INV-12345","2":123.45}}`}
	app := &App{LLM: m, Client: &mockPaperlessClient{CustomFields: []CustomField{{ID: 1, Name: "Invoice Number", DataType: "string"}, {ID: 2, Name: "Amount", DataType: "float"}, {ID: 3, Name: "Secret", DataType: "string"}}}}
	request := GenerateSuggestionsRequest{GenerateTitles: true, GenerateTags: true, GenerateCorrespondents: true, GenerateDocumentTypes: true, GenerateCreatedDate: true, GenerateCustomFields: true}
	doc := Document{ID: 1, Content: "Unique document body INV-12345", Tags: []string{"invoice"}}
	got, err := app.generateSingleDocumentSuggestion(context.Background(), request, doc, suggestionGenerationContext{availableTagNames: []string{"invoice"}, availableCorrespondentNames: []string{"Vendor"}, availableDocumentTypeNames: []string{"Invoice"}}, logrus.NewEntry(log))
	require.NoError(t, err)
	assert.Equal(t, 1, m.calls)
	assert.Equal(t, 1, strings.Count(m.lastPrompt, doc.Content))
	assert.Equal(t, "ACME invoice", got.SuggestedTitle)
	assert.Equal(t, []string{"invoice"}, got.SuggestedTags)
	assert.Equal(t, "Invoice", got.SuggestedDocumentType)
	assert.Equal(t, "2023-01-01", got.SuggestedCreatedDate)
	require.Len(t, got.SuggestedCustomFields, 2)
	spec, _ := m.tool.ParamsOneOf.ToJSONSchema()
	b, _ := json.Marshal(spec)
	assert.NotContains(t, string(b), "Secret")
}
func TestMetadataNoEnabledFieldsMakesNoRequest(t *testing.T) {
	app := &App{}
	doc := Document{ID: 1, Title: "original", Tags: []string{"a"}}
	got, err := app.generateSingleDocumentSuggestion(context.Background(), GenerateSuggestionsRequest{}, doc, suggestionGenerationContext{}, logrus.NewEntry(log))
	require.NoError(t, err)
	assert.Equal(t, doc.Title, got.SuggestedTitle)
	assert.Equal(t, doc.Tags, got.SuggestedTags)
}
func TestMetadataWorkflowInstructionsAndFlags(t *testing.T) {
	setupMetadataTest(t)
	enabled := true
	wf := &WorkflowConfig{ID: "special", GenerateTitles: &enabled, Prompts: map[string]string{"title_prompt": "Workflow title for {{.Content}}"}}
	m := &mockLLM{Response: `{"title":"workflow title"}`}
	app := &App{LLM: m, Client: &mockPaperlessClient{TagsError: fmt.Errorf("unrequested tags")}}
	req := GenerateSuggestionsRequest{Workflow: wf, Documents: []Document{{ID: 1, Content: "workflow content"}}}
	got, err := app.generateDocumentSuggestions(context.Background(), req, logrus.NewEntry(log))
	require.NoError(t, err)
	require.Len(t, got, 1)
	assert.Equal(t, "workflow title", got[0].SuggestedTitle)
	assert.Contains(t, m.lastPrompt, "Workflow title")
	assert.Equal(t, 1, m.calls)
}
func TestMetadataBudgetIncludesInstructionsAndTool(t *testing.T) {
	setupMetadataTest(t)
	tokenLimit = 500
	m := &mockLLM{Response: `{"title":"short"}`}
	app := &App{LLM: m}
	_, err := app.generateSingleDocumentSuggestion(context.Background(), GenerateSuggestionsRequest{GenerateTitles: true}, Document{Content: strings.Repeat("document word ", 10000)}, suggestionGenerationContext{}, logrus.NewEntry(log))
	require.NoError(t, err)
	spec, _ := m.tool.ParamsOneOf.ToJSONSchema()
	b, _ := json.Marshal(spec)
	promptTokens, err := getTokenCount(m.lastPrompt + string(b))
	require.NoError(t, err)
	assert.LessOrEqual(t, promptTokens, tokenLimit)
}
func TestMetadataRejectsUnexpectedOrInvalidFields(t *testing.T) {
	for _, response := range []string{`{"title":"ok","unexpected":"bad"}`, `{"title":123}`, `{}`, `{"title":"ok","custom_fields":{"99":"bad"}}`} {
		t.Run(response, func(t *testing.T) {
			setupMetadataTest(t)
			app := &App{LLM: &mockLLM{Response: response}}
			_, err := app.generateSingleDocumentSuggestion(context.Background(), GenerateSuggestionsRequest{GenerateTitles: true}, Document{}, suggestionGenerationContext{}, logrus.NewEntry(log))
			require.Error(t, err)
		})
	}
}
func TestMetadataDocumentLinksResolveOutsideModel(t *testing.T) {
	setupMetadataTest(t)
	settings.CustomFieldsSelectedIDs = []int{7}
	m := &mockLLM{Response: `{"custom_fields":{"7":["R10927801","V-2026-17","12","Law"]}}`}
	client := &mockPaperlessClient{CustomFields: []CustomField{{ID: 7, Name: "Related", DataType: documentLinkFieldType}}, ReferenceMatches: map[string][]int{"R10927801": {533, 412}, "V-2026-17": {412, 87}}}
	app := &App{LLM: m, Client: client}
	got, err := app.generateSingleDocumentSuggestion(context.Background(), GenerateSuggestionsRequest{GenerateCustomFields: true}, Document{ID: 533}, suggestionGenerationContext{}, logrus.NewEntry(log))
	require.NoError(t, err)
	require.Len(t, got.SuggestedCustomFields, 1)
	assert.Equal(t, []int{412, 87}, got.SuggestedCustomFields[0].Value)
	assert.Equal(t, []string{"R10927801", "V-2026-17"}, client.ReferenceLookups)
}
func TestMetadataCustomTypesAndOmittedValues(t *testing.T) {
	setupMetadataTest(t)
	settings.CustomFieldsSelectedIDs = []int{1, 2, 3}
	client := &mockPaperlessClient{CustomFields: []CustomField{{ID: 1, Name: "Amount", DataType: "monetary"}, {ID: 2, Name: "Date", DataType: "date"}, {ID: 3, Name: "Missing", DataType: "string"}}}
	app := &App{LLM: &mockLLM{Response: `{"custom_fields":{"1":"1.234,56 EUR","2":"2023.01.01"}}`}, Client: client}
	got, err := app.generateSingleDocumentSuggestion(context.Background(), GenerateSuggestionsRequest{GenerateCustomFields: true}, Document{}, suggestionGenerationContext{}, logrus.NewEntry(log))
	require.NoError(t, err)
	require.Len(t, got.SuggestedCustomFields, 2)
	assert.Equal(t, "EUR1234.56", got.SuggestedCustomFields[0].Value)
	assert.Equal(t, "2023-01-01", got.SuggestedCustomFields[1].Value)
}
func TestFilterSuggestedTagsPreservesExistingAndExcludesSystem(t *testing.T) {
	oldCreate := createNewTags
	oldAuto := autoTag
	t.Cleanup(func() { createNewTags = oldCreate; autoTag = oldAuto })
	autoTag = "paperless-gpt-auto"
	for _, create := range []bool{false, true} {
		createNewTags = create
		got := filterSuggestedTags([]string{"INVOICE", "new", "paperless-gpt-auto"}, []string{"invoice", "Insurance", "paperless-gpt-auto"}, []string{"Insurance", "paperless-gpt-auto"})
		assert.Contains(t, got, "invoice")
		assert.Contains(t, got, "Insurance")
		assert.NotContains(t, got, "paperless-gpt-auto")
		assert.Equal(t, create, slices.Contains(got, "new"))
	}
}
