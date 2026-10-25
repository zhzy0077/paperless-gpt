# Mock LLM stubs (WireMock)

These mappings return result-tool calls for the Eino document-processing path.
`title.json` handles a combined metadata request. Its response template emits
exactly the requested fields and selected custom-field IDs from the tool Schema.
The workflow mappings use prompt markers to return deterministic workflow values.
`ocr.json` returns a result tool with the recognized text. `default.json` serves
free-form ad-hoc analysis.

Run `npm run test:e2e:mock` to exercise the UI, application and Paperless API with
these stubs. The test environment enables WireMock's response-template transformer
for these mappings, so changing generation flags changes the response fields.
