# Error Detection Implementation Status

## What Was Implemented

### Files Created
1. **`llm_client.py`** - Portkey client wrapper for LLM calls with retry logic
2. **`prompts/error_detection.yaml`** - LLM prompts for L2 checks (using `openai/gpt-4o-mini`)
3. **`error_detector.py`** - Main error detection module with:
   - `trace_errors` table initialization
   - 14 Level 1 checks (no LLM)
   - 3 Level 2 checks (LLM-based)
   - Multi-processing support via `--workers` flag
   - Batch processing for dashboard stats

### Files Modified
1. **`config.py`** - Added `ERROR_DETECTION_PROMPTS` path
2. **`analyzer.py`** - Added error query functions and filter support
3. **`components.py`** - Added `ErrorCheckResultsPanel`, `ErrorDetectionStatsPanel`, `ErrorDetectionFilterPanel`
4. **`main.py`** - Added routes `/run-error-checks`, `/run-error-check/{element_id}`, `/error-detection`
5. **`requirements.txt`** - Added `portkey-ai`, `python-dotenv`, `pyyaml`

---

## Current Status

### L1 Checks (Complete for all 28,628 traces)
| Check | Errors | Total | Rate |
|-------|--------|-------|------|
| L1_INVALID_ACTION_TYPE | 364 | 3,359 | 10.8% |
| L1_TOO_SHORT | 293 | 364 | 80.5% |
| L1_EMPTY_RESPONSE | 180 | 28,628 | 0.6% |
| L1_MISSING_SECTIONS | 54 | 364 | 14.8% |
| L1_UNBALANCED_HIGHLIGHT | 10 | 24,697 | 0.0% |
| L1_REFUSAL | 2 | 28,628 | 0.0% |

### L2 Checks (Partial - ~940 traces checked)
| Check | Errors | Total | Rate |
|-------|--------|-------|------|
| L2_FORMAT_ERROR | 53 | 577 | 9.2% |
| L2_HALLUCINATION | 12 | 939 | 1.3% |
| L2_CLINICAL_SAFETY | 2 | 203 | 1.0% |

---

## What Needs to be Done

### Run L2 checks for all use cases with 50 workers:

```bash
# 1. patient_summary (364 traces) - ~30 sec
python error_detector.py --use-case worklist_patient_summary --level2 --workers 50

# 2. suggest_next_steps (3,476 traces) - ~3 min
python error_detector.py --use-case worklist_suggest_next_steps --level2 --workers 50

# 3. rad_report_data_extraction (24,788 traces) - ~20 min
python error_detector.py --use-case worklist_rad_report_data_extraction --level2 --workers 50
```

**Total estimated time: ~25 minutes** with `gpt-4o-mini` and 50 workers

### Or run all at once (sequential):
```bash
python error_detector.py --use-case worklist_patient_summary --level2 --workers 50 && \
python error_detector.py --use-case worklist_suggest_next_steps --level2 --workers 50 && \
python error_detector.py --use-case worklist_rad_report_data_extraction --level2 --workers 50
```

---

## Use Case to Prompt Hash Mapping

```python
USE_CASE_HASHES = {
    '61422e2ac4ee4efb': 'worklist_rad_report_data_extraction',
    '9a3acb64787e8a9f': 'worklist_rad_report_data_extraction',
    '995558dcc87664fa': 'worklist_suggest_next_steps',
    'aec9115a1ea76ab0': 'worklist_patient_summary',
}
```

---

## L2 Checks per Use Case

| Use Case | L2 Checks |
|----------|-----------|
| patient_summary | L2_HALLUCINATION |
| suggest_next_steps | L2_HALLUCINATION, L2_FORMAT_ERROR, L2_CLINICAL_SAFETY |
| rad_report_data_extraction | L2_HALLUCINATION, L2_FORMAT_ERROR |

---

## CLI Usage

```bash
# Run L1 only
python error_detector.py --use-case worklist_patient_summary

# Run L1 + L2 with workers
python error_detector.py --use-case worklist_patient_summary --level2 --workers 50

# Check stats
python error_detector.py --stats

# Run checks on single element
python error_detector.py --element <element_id> --level2
```

---

## Environment Requirements

`.env` file needs:
```
PORTKEY_API_KEY=your_portkey_api_key
PROVIDER_NAME=your_virtual_key
```

---

## Dashboard URLs

- `/` - Main dashboard with error detection stats panel
- `/error-detection` - Detailed error detection page
- `/traces?error_status=has_errors` - Filter traces with errors
- `/trace/{trace_id}` - Individual trace view with error check results panel
