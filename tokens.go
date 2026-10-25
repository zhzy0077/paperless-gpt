package main

import (
	"fmt"
	"sync"

	"github.com/pkoukk/tiktoken-go"
)

func getTokenCount(content string) (int, error) {
	tokenizerOnce.Do(func() { tokenizer, tokenizerErr = tiktoken.GetEncoding("cl100k_base") })
	if tokenizerErr != nil {
		return 0, tokenizerErr
	}
	return len(tokenizer.Encode(content, nil, nil)), nil
}

// truncateContentByTokens truncates the content so that its token count does not exceed availableTokens.
// This implementation uses a binary search on runes to find the longest prefix whose token count is within the limit.
// If availableTokens is 0 or negative, the original content is returned.
func truncateContentByTokens(content string, availableTokens int) (string, error) {
	if availableTokens < 0 || tokenLimit <= 0 {
		return content, nil
	}
	totalTokens, err := getTokenCount(content)
	if err != nil {
		return "", fmt.Errorf("error counting tokens: %v", err)
	}
	if totalTokens <= availableTokens {
		return content, nil
	}

	// Convert content to runes for safe slicing.
	runes := []rune(content)
	low := 0
	high := len(runes)
	validCut := 0

	for low <= high {
		mid := (low + high) / 2
		substr := string(runes[:mid])
		count, err := getTokenCount(substr)
		if err != nil {
			return "", fmt.Errorf("error counting tokens in substring: %v", err)
		}
		if count <= availableTokens {
			validCut = mid
			low = mid + 1
		} else {
			high = mid - 1
		}
	}

	truncated := string(runes[:validCut])
	// Final verification
	finalTokens, err := getTokenCount(truncated)
	if err != nil {
		return "", fmt.Errorf("error counting tokens in final truncated content: %v", err)
	}
	if finalTokens > availableTokens {
		return "", fmt.Errorf("truncated content still exceeds the available token limit")
	}
	return truncated, nil
}

var tokenizerOnce sync.Once
var tokenizer *tiktoken.Tiktoken
var tokenizerErr error
