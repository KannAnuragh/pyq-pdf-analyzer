# Malayalam Question Correction & Validation — Final Plan (Unconditional Node)

This plan implements the final architecture for the correction & validation pipeline, superseding the previous gated/separate node design. Verification is unconditional on every question, correction happens inside that same call, and the cache permanently patches the extraction layer without gatekeeping the LLM call.

## User Review Required

> [!WARNING]
> This plan completely removes the old `solve_and_validate_node` and its fallback to web search, consolidating both text correction and answer verification into a single, unconditional LLM call (`correction_and_validation_node`).

## Open Questions

- What should be the default text_confidence value when skipping corrections (should it be 1.0 since it's unflagged)?
- Do we still need to compute token usage for the summary?

## Proposed Changes

---

### Graph Architecture

#### [MODIFY] [graph/workflow.py](file:///home/anuragh/Trogon/qpaper%20scanner/pdf_qa_agent/graph/workflow.py)
- Remove `solve_and_validate_node` from the graph.
- Update edges: `rule_check` routes to `END`.
- The sequence will be: `extract` -> `quality_score` -> `repair` -> `correction_and_validation` -> `rule_check` -> `END`.

#### [MODIFY] [graph/nodes.py](file:///home/anuragh/Trogon/qpaper%20scanner/pdf_qa_agent/graph/nodes.py)
- Delete `solve_and_validate_node` and its helper `_solve_single_block`.
- Ensure `rule_check_node` correctly handles `processed_questions` populated by `correction_and_validation_node`.

---

### Unconditional Correction & Validation Node

#### [MODIFY] [graph/correction_node.py](file:///home/anuragh/Trogon/qpaper%20scanner/pdf_qa_agent/graph/correction_node.py)
- **Unconditional Execution**: Remove the check that skips the LLM if there are no flagged words. The LLM must be called for all questions.
- **State Output**: Update `correction_and_validation_node` to output `processed_questions` (with `validated_answer`, `is_valid`, `reasoning`) to fully replace the validation step.
- **Prompting**: Update `_build_page_prompt` and `CORRECTION_SYSTEM_PROMPT` to include all questions. Unflagged questions will instruct the LLM to skip text correction and only perform answer verification.

#### [MODIFY] [ingest/word_router.py](file:///home/anuragh/Trogon/qpaper%20scanner/pdf_qa_agent/ingest/word_router.py)
- Review to ensure it correctly identifies flagged words and applies Aho-Corasick pre-substitution (already implemented). No major changes expected here as it currently correctly returns `RoutingResult`.

## Verification Plan

### Automated Tests
- N/A

### Manual Verification
- Upload a test PDF.
- Verify that `correction_and_validation_node` processes all questions regardless of flagged words.
- Verify that `processed_questions` contains the expected validation results and corrections.
