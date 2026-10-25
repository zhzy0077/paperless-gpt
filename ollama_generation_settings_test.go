package main

import (
	"encoding/json"
	"github.com/stretchr/testify/require"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

func floatPointer(v float64) *float64 { return &v }
func intPointer(v int) *int           { return &v }
func stringPointer(v string) *string  { return &v }
func testOllamaServer(t *testing.T, h http.HandlerFunc) *httptest.Server {
	t.Helper()
	s := httptest.NewServer(h)
	t.Cleanup(s.Close)
	return s
}
func TestOllamaGenerationSettingsValidation(t *testing.T) {
	for _, s := range []OllamaGenerationSettings{{MaxTokens: intPointer(0)}, {MaxTokens: intPointer(-2)}, {ContextLength: intPointer(0)}, {Temperature: floatPointer(math.NaN())}, {Temperature: floatPointer(-1)}, {TopP: floatPointer(1.1)}, {MinP: floatPointer(-0.1)}, {KeepAlive: stringPointer("bad")}, {Think: json.RawMessage(`"invalid"`)}, {Format: json.RawMessage(`[]`)}, {RepeatLastN: intPointer(-2)}} {
		require.Error(t, s.Validate())
	}
	for _, s := range []OllamaGenerationSettings{{}, {MaxTokens: intPointer(-1), Temperature: floatPointer(0), KeepAlive: stringPointer("-1"), Think: json.RawMessage(`false`)}, {Think: json.RawMessage(`"high"`), Format: json.RawMessage(`{"type":"object"}`)}} {
		require.NoError(t, s.Validate())
	}
}
