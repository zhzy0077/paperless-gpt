package llm

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
	"golang.org/x/time/rate"
)

type Policy struct {
	RequestsPerMinute float64
	MaxRetries        int
	BackoffMaxWait    time.Duration
}
type retryModel struct {
	model   Model
	policy  Policy
	limiter *rate.Limiter
}

func WithPolicy(m Model, p Policy) Model {
	r := &retryModel{model: m, policy: p}
	if p.RequestsPerMinute > 0 {
		r.limiter = rate.NewLimiter(rate.Limit(p.RequestsPerMinute/60), 1)
	}
	return r
}
func (r *retryModel) Generate(ctx context.Context, in []*schema.Message, opts ...model.Option) (*schema.Message, error) {
	maxWait := r.policy.BackoffMaxWait
	if maxWait <= 0 {
		maxWait = 30 * time.Second
	}
	for attempt := 0; ; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if r.limiter != nil {
			if err := r.limiter.Wait(ctx); err != nil {
				return nil, err
			}
		}
		out, err := r.model.Generate(ctx, in, opts...)
		if err == nil {
			return out, nil
		}
		if attempt >= r.policy.MaxRetries || !IsTransient(err) {
			return nil, err
		}
		delay := time.Second
		for i := 0; i < attempt && delay < maxWait; i++ {
			if delay > maxWait/2 {
				delay = maxWait
				break
			}
			delay *= 2
		}
		if delay > maxWait {
			delay = maxWait
		}
		delay = time.Duration(float64(delay) * (0.8 + rand.Float64()*0.2))
		delay = RetryDelay(err, delay)
		if delay > 15*time.Minute {
			return nil, fmt.Errorf("provider retry delay %s exceeds limit", delay)
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}
func RetryDelay(err error, backoff time.Duration) time.Duration {
	var retry interface{ RetryAfter() time.Duration }
	if errors.As(err, &retry) && retry.RetryAfter() > backoff {
		backoff = retry.RetryAfter()
	}
	const marker = "retry-after-ms:"
	s := strings.ToLower(err.Error())
	if i := strings.Index(s, marker); i >= 0 {
		s = s[i+len(marker):]
		n := 0
		for n < len(s) && s[n] >= '0' && s[n] <= '9' {
			n++
		}
		if ms, e := strconv.ParseInt(s[:n], 10, 64); e == nil && ms >= 0 && ms <= int64((1<<63-1)/time.Millisecond) {
			if d := time.Duration(ms) * time.Millisecond; d > backoff {
				backoff = d
			}
		}
	}
	return backoff
}
