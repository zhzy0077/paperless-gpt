# Eino migration review

The LLM module has been rewritten behind the existing application contracts. The following review records its measured scope and implementation plan. Put provider access behind Eino, combine metadata extraction into one structured request, and keep Paperless integration and workflow execution in the application layer.

## Measured scope

The repository has two external integration boundaries: Paperless and model providers. Between them it implements document processing, workflow execution, review approval, jobs, history, and PDF lifecycle management.

A Go lexical scan counts lines containing source tokens, excluding comments, blank lines, and automatically inserted semicolons. This measurement is a source-code baseline; the reduction ranges below are engineering estimates.

| Area | Production Go code lines |
|---|---:|
| Entire Go backend | 11,891 |
| LLM migration scope | 2,535 |
| Paperless implementation (`paperless.go`) | 1,622 |
| OCR orchestration (`ocr.go`) | 655 |
| Background processing (`background.go`) | 522 |

There are 49 production Go files and 49 Go test files. Tests contain 10,749 code lines. Eight production files and fourteen test files reference `langchaingo`.

The 2,535-line migration scope consists of `app_llm.go`, `llm_client.go`, `app_llm_googleai.go`, `ollama_metadata.go`, `ollama_config.go`, `ollama_generation_settings.go`, `ocr/llm_provider.go`, `ocr/google_ai_client.go`, `tokens.go`, and the two model-factory functions in `main.go`. It represents 21.3% of backend production code. It includes provider configuration and application rules that remain necessary after migration.

| Existing component | Code lines | Disposition |
|---|---:|---|
| `app_llm.go` | 912 | Replace six generation paths with one extraction; retain field selection, normalization, candidate filtering and document-link handling |
| `llm_client.go` | 160 | Consolidate limits, cancellation and transient-error retries into one policy |
| `ollama_metadata.go` | 298 | Replace native-client adapter with the Eino adapter; migrate the request semantics |
| `ollama_config.go` + `ollama_generation_settings.go` | 266 | Keep configuration compatibility and validation; remove library-specific conversion |
| `app_llm_googleai.go` + `ocr/google_ai_client.go` | 253 | Replace SDK-to-langchaingo conversion with Eino adapters |
| `ocr/llm_provider.go` | 383 | Keep the OCR Provider bridge and HTTP policies; delegate model access and result extraction |
| `tokens.go` | 74 | Replace the tokenizer dependency separately from the model client |
| `createLLM` + `createVisionLLM` | 189 | One provider factory, parameterized by metadata/vision configuration |

## Findings that determine the cost

1. **An unused vision client is initialized at startup.** `App.VisionLLM` is assigned but production OCR uses `app.ocrProvider` and its own client. Remove the unused field and `createVisionLLM` path while migrating, eliminating a second initialization path.
2. **The six metadata functions duplicate prompt rendering, truncation, model requests, and text parsing.** They account for 518 code lines before shared helpers and orchestration. The new path builds one requested-field Schema and returns one result.
3. **The application already has a Paperless seam.** `ClientInterface` in `types.go` allows the LLM rewrite to preserve document retrieval and updates. `DocumentSuggestion` and existing HTTP payloads remain the integration contract.
4. **LLM code currently calls Paperless itself.** Candidate lookup and document-link resolution live inside `app_llm.go`. Fetch candidates before extraction and resolve extracted references afterward, in the application layer. The extraction package receives data and returns data.
5. **Prompts are persisted configuration.** Six per-task prompt files, workflow overrides, template variables and the prompt editor are part of the migration. Later prompts currently receive the generated title. Combined extraction requires an explicit replacement for that staged dependency.
6. **There are three model operations.** Metadata is structured extraction, OCR accepts image/PDF content, and ad-hoc analysis returns free-form text. The new module supports each operation through the same model-access layer.
7. **HTTP and retry behavior is application functionality.** Preserve timeouts, caller cancellation, Retry-After, custom headers and the OpenRouter request policy. Assign retry ownership to one layer to keep attempts bounded. A token-exhausted result is a generation failure with usage recorded.
8. **Provider-specific settings remain part of configuration.** Examples include Ollama context length, keep-alive and thinking values; Gemini thinking budget; Azure API settings; and independent metadata/vision credentials and limits. Translate these centrally into Eino adapter options.
9. **Document processing extends beyond the model call.** Keep page selection, searchable PDF generation, version uploads, run records and tagging in the current OCR workflow. Dedicated OCR services (Docling, Document AI, Azure Document Intelligence, Mistral OCR) continue to use their existing OCR provider interface.
10. **Removing langchaingo also requires replacing token counting and test doubles.** `tokens.go` and OCR fallback token counting use `llms.CountTokens`; fourteen test files reference the library. Change mocks to the new small interfaces and retain behavior-oriented tests.

## Proposed boundary

```text
HTTP handlers / background jobs / workflow execution
                  |
         Paperless ClientInterface
                  |
          Extraction input builder
                  |
      internal/extraction: metadata / OCR / analysis
                  |
      internal/llm: Eino clients + request policy
                  |
                 API
```

The application supplies document text, images, candidate names, selected field definitions and workflow instructions. The extractor returns typed suggestions and usage. Paperless writes, reference searches, tag handover and history are owned by the application.

Suggested layout:

```text
internal/llm/
  client.go       Eino calls, text and media messages, tool arguments
  providers.go    Adapter factory and provider-specific options
  policy.go       Timeouts, limits, retry classification
internal/extraction/
  metadata.go     Requested-field tool and one-call extraction
  prompts.go      Prompt rendering and migration helpers
  validation.go   Allowed values, field types, missing/empty semantics
ocr/llm_provider.go
                  Existing Provider interface delegating to the new module
```

Keep independent metadata and vision configuration. Use immutable `WithTools` instances for concurrent requests. Tool parameters carry the Schema; prompts describe the task. For the tested DeepSeek-compatible endpoint, explicitly send `thinking.type=disabled` and force the result tool. Native provider adapters receive their corresponding thinking settings.

Requested metadata fields determine the tool Schema. Disabled fields are omitted. Preserve the difference between absent results, explicit empty results and invalid results. Continue merging existing tags and enforcing the system-tag exclusions in the application. Keep monetary/date normalization, select-option mapping and document-reference resolution.

The Eino factory selects adapters by the actual protocol and required capabilities. Our live test confirms the OpenAI-compatible adapter with images, named tools and disabled thinking. Native provider coverage is verified with captured-request tests for its adapter options, including Ollama-specific settings.

## Cost and code-size estimate

Estimate for one engineer familiar with this repository, preserving current providers, environment settings, workflow behavior and HTTP contracts. Engineering days are approximately eight focused hours.

| Scope | Estimated effort | Result |
|---|---:|---|
| Replace connector implementation only | 2–3 days | Eino model access; existing six-call metadata path and prompt machinery remain |
| Rewrite the LLM module and unify extraction — recommended | 6–9 days | Shared Eino access, one-call metadata, integrated OCR/analysis, migrated configuration/prompts and regression coverage |

Breakdown of the recommended scope:

| Work | Estimated days |
|---|---:|
| Lock contracts, build fixtures and record baseline behavior | 0.5 |
| Eino factory, settings mapping and HTTP/request policy | 1–1.5 |
| Unified metadata extraction and local validation | 1–1.5 |
| OCR and ad-hoc analysis integration; remove unused vision path | 0.5–1 |
| Prompt/workflow migration and configuration/UI consistency | 1–1.5 |
| Rewrite affected tests, provider wire checks, E2E and cleanup | 2–3 |
| **Total** | **6–9** |

Target the 2,535-line scope at approximately 1,200–1,600 production code lines. This is an estimated net reduction of about 900–1,300 lines: 37–53% of the affected module and approximately 8–11% of the backend. Moving existing functions into a new package does not count as a reduction. The main savings come from provider conversions, duplicated extraction paths and obsolete response parsers.

Prompt compatibility is the largest design item. Preserve current prompt files and workflow overrides as migration inputs, introduce a unified metadata template, and make migration status visible in the existing prompt/workflow UI. Archive customized originals. A permanent second implementation of the old extraction path adds code and maintenance, so use an explicit migration step with a documented instruction-combination rule.

## Implementation sequence

1. Add the small model and extraction contracts; create provider request fixtures from existing tests and the successful live Eino integration.
2. Implement Eino provider construction and a shared request policy. Establish metadata, image OCR and analysis calls through that layer.
3. Build the requested-field tool and one-call metadata extractor. Keep the current DocumentSuggestion assembly and Paperless update path.
4. Migrate global and workflow prompts; update prompt defaults/editor metadata together. Migrate provider settings without changing their documented meaning.
5. Move the LLM OCR provider and ad-hoc analysis to the shared layer, then remove the unused VisionLLM initialization.
6. Remove langchaingo, obsolete adapters and old parsers after all production call sites and tests use the new interfaces. Replace token counting with a maintained tokenizer behind the existing budget helper.
7. Build frontend assets, run the backend suite and formatting/lint checks, run mock-LLM E2E, and repeat the real-endpoint PDF evaluation.

## Acceptance criteria

- One model request for the enabled metadata fields of one document; no request if all generation fields are disabled.
- Real endpoint: three image-OCR runs and three metadata runs produce valid result tool arguments with thinking disabled. The receipt retains 21/21 item tuples and metadata passes the existing 17 checks.
- Wire tests verify image bytes/MIME type, tool choice, Schema, thinking control and usage across the provider configurations being retained.
- Existing workflow flags, original tags, system tags, selected custom fields, candidate matching, normalization, document links, approval, undo and partial Paperless updates retain their tested behavior.
- Cancelled requests terminate; transient requests respect bounded retries and Retry-After; exhausted generations surface as failed/incomplete results with usage.
- Prompt/template migrations preserve customized input and remain usable through the prompt and workflow editors.
- The backend imports no langchaingo package. Code-size reporting compares the same production-file scope before and after.

## Review evidence

Implemented the Eino model boundary, combined metadata extraction, structured OCR results and free-form analysis. The Paperless update implementation and HTTP result contracts are retained. Source-count details are retained in the local evaluation artifacts as `migration-review/go-source-counts.json`.

The prior standalone live Eino integration is documented in the local evaluation artifacts under `eino-integration/README.md`. Its pinned modules are Eino v0.7.13, the compatible adapter v0.1.13 and Sonic v1.15.4; its final nine acceptance requests passed. The implemented provider wire tests cover OpenAI-compatible images, native Claude tools, Gemini PDF input, native Ollama constraints, and concurrent native options. Full Docker E2E requires a container runtime.

## Implemented result

| Measurement | Before | After |
|---|---:|---:|
| LLM production scope | 2,535 | 1539 |
| Backend production code | 11,891 | 10898 |

The LLM scope is reduced by 996 lines (39.3%); backend production code is reduced by 993 lines (8.4%). Counts use the same Go-token scan as the baseline, including the new shared module and moved HTTP helpers. Test code is counted separately.

The implementation uses `internal/llm` for Eino access and keeps metadata assembly/reference resolution in `app_llm.go`. It removes the old framework dependency and provider conversion files, plus the unused startup VisionLLM client. Existing per-field templates are rendered as instruction sections; document content is shared by the combined request. Default prompt history upgrades older uncustomized files. Native Ollama uses schema-constrained output normalized into the result-tool contract; compatible, Claude and Gemini adapters use tool calls.

Configuration changes: extraction defaults to zero temperature on supported models; GPT-5/o-series requests use max_completion_tokens with the model's fixed sampling, and newer Claude models omit temperature. DeepSeek-compatible models disable thinking by default. LLM_MAX_TOKENS now applies to hosted metadata/analysis as well as native Ollama. Hosted metadata/analysis and LLM OCR default to 8192 output tokens; native Ollama metadata retains its configured/model budget. Existing OLLAMA_THINK and GOOGLEAI_THINKING_BUDGET settings are preserved. Truncated generations fail with usage instead of becoming blank OCR pages.

Validation artifacts are retained in the local evaluation directory under `rewritten-module/`. Frontend build succeeds. Lint reports the same 21 existing findings by file/rule as HEAD. All 252 metadata-field combinations were exercised against actual WireMock 3.13.1. Application startup, API routes and embedded frontend assets were verified on port 8080. The Docker-backed Playwright suite reaches setup and stops because this environment has no container runtime.

To repeat the opt-in receipt benchmark, set OPENAI_API_KEY, OPENAI_BASE_URL, EINO_EVAL_IMAGE, EINO_EVAL_GOLD and EINO_EVAL_MODEL, then run `go test -run '^TestLiveEinoMigration$' -count=1 -v .`. The golden file contains the receipt facts and name/quantity/unit tuples; ordinary tests skip this live request.

Final acceptance: `go test ./...` passes; the live receipt evaluation passes three runs with 21/21 item tuples, the expected metadata, and all ten selected custom fields. The shared model package is checked with the race detector. The production build and formatting checks pass.
