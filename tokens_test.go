package main

import (
	"os"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// resetTokenLimit parses TOKEN_LIMIT from environment and sets the tokenLimit variable
func resetTokenLimit() {
	// Reset tokenLimit
	tokenLimit = 0
	// Parse from environment
	if limit := os.Getenv("TOKEN_LIMIT"); limit != "" {
		if parsed, err := strconv.Atoi(limit); err == nil {
			tokenLimit = parsed
		}
	}
}

func TestTokenLimit(t *testing.T) {
	// Save current env and restore after test
	originalLimit := os.Getenv("TOKEN_LIMIT")
	defer os.Setenv("TOKEN_LIMIT", originalLimit)

	tests := []struct {
		name      string
		envValue  string
		wantLimit int
	}{
		{
			name:      "empty value",
			envValue:  "",
			wantLimit: 0,
		},
		{
			name:      "zero value",
			envValue:  "0",
			wantLimit: 0,
		},
		{
			name:      "positive value",
			envValue:  "1000",
			wantLimit: 1000,
		},
		{
			name:      "invalid value",
			envValue:  "not-a-number",
			wantLimit: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Set environment variable
			os.Setenv("TOKEN_LIMIT", tc.envValue)

			// Set tokenLimit based on environment
			resetTokenLimit()

			assert.Equal(t, tc.wantLimit, tokenLimit)
		})
	}
}

func TestTruncateContentByTokens(t *testing.T) {
	// Save current env and restore after test
	originalLimit := os.Getenv("TOKEN_LIMIT")
	defer os.Setenv("TOKEN_LIMIT", originalLimit)

	// Set a token limit for testing
	os.Setenv("TOKEN_LIMIT", "100")
	// Set tokenLimit based on environment
	resetTokenLimit()

	tests := []struct {
		name            string
		content         string
		availableTokens int
		wantTruncated   bool
		wantErr         bool
	}{
		{
			name:            "no truncation needed",
			content:         "short content",
			availableTokens: 20,
			wantTruncated:   false,
			wantErr:         false,
		},
		{
			name:            "disabled by token limit",
			content:         "any content",
			availableTokens: -1,
			wantTruncated:   false,
			wantErr:         false,
		},
		{
			name:            "truncation needed",
			content:         "This is a much longer content that will definitely need to be truncated because it exceeds the available tokens",
			availableTokens: 10,
			wantTruncated:   true,
			wantErr:         false,
		},
		{
			name:            "empty content",
			content:         "",
			availableTokens: 10,
			wantTruncated:   false,
			wantErr:         false,
		},
		{
			name:            "exact token count",
			content:         "one two three four five",
			availableTokens: 5,
			wantTruncated:   false,
			wantErr:         false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result, err := truncateContentByTokens(tc.content, tc.availableTokens)

			if tc.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)

			if tc.wantTruncated {
				assert.True(t, len(result) < len(tc.content), "Content should be truncated")
			} else {
				assert.Equal(t, tc.content, result, "Content should not be truncated")
			}
		})
	}
}
