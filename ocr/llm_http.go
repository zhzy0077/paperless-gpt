package ocr

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"paperless-gpt/internal/openrouterzdr"
	"strconv"
	"strings"
	"time"
)

const maxProviderRetryDelay = 15 * time.Minute

// ParseHeaderList parses a comma-separated list of Key=Value pairs, the
// format of OLLAMA_HEADERS and OPENAI_HEADERS. Spaces around keys and values
// are dropped ("X-Team = docs" would otherwise be an invalid header name that
// fails every request); malformed pairs are ignored.
func ParseHeaderList(raw string) map[string]string {
	headers := map[string]string{}
	for _, pair := range strings.Split(raw, ",") {
		key, value, ok := strings.Cut(pair, "=")
		key = http.CanonicalHeaderKey(strings.TrimSpace(key))
		if ok && key != "" {
			headers[key] = strings.TrimSpace(value)
		}
	}
	return headers
}

// OllamaHTTPClient returns an *http.Client with headers from OLLAMA_HEADERS injected,
// or nil if OLLAMA_HEADERS is not set.
func OllamaHTTPClient() *http.Client {
	headers := ParseHeaderList(os.Getenv("OLLAMA_HEADERS"))
	if len(headers) == 0 {
		return nil
	}
	return headerClient(headers)
}

// OpenAIHTTPClient returns the client for every OpenAI-compatible request:
// text LLM, vision LLM and LLM OCR. It sends X-Title and the OPENAI_HEADERS.
// It is a client of its own on purpose: those headers can carry gateway
// credentials and must never reach other hosts through http.DefaultClient.
func OpenAIHTTPClient() *http.Client {
	headers := map[string]string{http.CanonicalHeaderKey("X-Title"): "paperless-gpt"}
	for key, value := range ParseHeaderList(os.Getenv("OPENAI_HEADERS")) {
		headers[key] = value
	}
	client := headerClient(headers)
	// OPENROUTER_ENFORCE_ZDR: a passthrough unless enabled and the request goes
	// to openrouter.ai; see internal/openrouterzdr. It sits below the header
	// transport, so it covers the text LLM, the vision LLM and LLM OCR alike.
	client.Transport.(*headerTransport).base = openrouterzdr.NewTransport(http.DefaultTransport)
	return client
}

// headerClient adds headers to every request. The headers are configured for
// one endpoint and may be credentials, so the client refuses to follow a
// redirect to another host: the transport would add them there too.
func headerClient(headers map[string]string) *http.Client {
	return &http.Client{
		Transport: &headerTransport{base: http.DefaultTransport, headers: headers},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return errors.New("stopped after 10 redirects")
			}
			if req.URL.Host != via[0].URL.Host {
				return fmt.Errorf("refusing to follow a redirect from %s to %s with configured headers", via[0].URL.Host, req.URL.Host)
			}
			return nil
		},
	}
}

type headerTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func parseRetryAfter(value string, now time.Time) (time.Duration, bool) {
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil && seconds >= 0 {
		maxSeconds := int64((1<<63 - 1) / int64(time.Second))
		if seconds <= maxSeconds {
			return time.Duration(seconds) * time.Second, true
		}
		return 0, false
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0, false
	}
	if delay := when.Sub(now); delay > 0 {
		// Preserve the provider's lower bound when the transport serializes
		// this duration to milliseconds; truncation could retry early.
		whole := delay / time.Millisecond
		if delay%time.Millisecond != 0 {
			const maxWholeMilliseconds = int64((1<<63 - 1) / int64(time.Millisecond))
			if int64(whole) < maxWholeMilliseconds {
				whole++
			}
		}
		return time.Duration(whole) * time.Millisecond, true
	}
	return 0, true
}

func retryAfterDelay(err error, backoff time.Duration) time.Duration {
	const marker = "retry-after-ms:"
	text := strings.ToLower(err.Error())
	index := strings.Index(text, marker)
	if index < 0 {
		return backoff
	}
	value := text[index+len(marker):]
	end := 0
	for end < len(value) && value[end] >= '0' && value[end] <= '9' {
		end++
	}
	if end == 0 {
		return backoff
	}
	milliseconds, parseErr := strconv.ParseInt(value[:end], 10, 64)
	maxMilliseconds := int64((1<<63 - 1) / int64(time.Millisecond))
	if parseErr != nil || milliseconds < 0 || milliseconds > maxMilliseconds {
		return backoff
	}
	delay := time.Duration(milliseconds) * time.Millisecond
	if delay > backoff {
		return delay
	}
	return backoff
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	resp, err := t.base.RoundTrip(req)
	if err != nil || resp == nil || resp.StatusCode != http.StatusTooManyRequests {
		return resp, err
	}
	delay, ok := parseRetryAfter(resp.Header.Get("Retry-After"), time.Now())
	if !ok {
		return resp, nil
	}
	_ = resp.Body.Close()
	// Provider SDKs can discard response headers from errors. Preserve only
	// the parsed cooldown in a sanitized JSON error body so the shared retry policy can
	// honor it without exposing the upstream response or credentials.
	resp.Body = io.NopCloser(strings.NewReader(fmt.Sprintf(
		`{"error":{"message":"upstream rate limit; retry-after-ms:%d"}}`,
		delay.Milliseconds())))
	resp.ContentLength = int64(-1)
	return resp, nil
}
