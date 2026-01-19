# TraceAudit - Repository Workflow

## Overview
TraceAudit is an LLM trace analysis and visualization platform built with FastHTML. It processes LLM traces from JSONL files, extracts system prompts to identify unique use cases, performs comprehensive analytics, and presents the results through an interactive web dashboard.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TraceAudit                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────────┐     ┌────────────────────┐     │
│   │   data/      │     │ data_processor.py│     │     output/        │     │
│   │  .jsonl file │────▶│  (ETL Pipeline)  │────▶│  - traces.db       │     │
│   │  (50K traces)│     │                  │     │  - analysis_cache  │     │
│   └──────────────┘     └──────────────────┘     └────────────────────┘     │
│                               │                          │                 │
│                               │                          │                 │
│                               ▼                          │                 │
│                        ┌──────────────┐                  │                 │
│                        │  prompts/    │                  │                 │
│                        │ prompt_*.txt │                  │                 │
│                        └──────────────┘                  │                 │
│                                                          │                 │
│                                                          ▼                 │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │                        analyzer.py                                │    │
│   │              (Statistics & Analysis Engine)                       │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                    │                                       │
│                                    ▼                                       │
│   ┌──────────────────────────────────────────────────────────────────┐    │
│   │    main.py (FastHTML Web Server) + components.py (UI Components)  │    │
│   │                                                                    │    │
│   │    Routes:                                                         │    │
│   │    • /           → Dashboard (stats, charts, use case overview)    │    │
│   │    • /use-cases  → All use cases listing                           │    │
│   │    • /use-case/{hash} → Use case detail with sample traces         │    │
│   │    • /trace/{id} → Individual trace viewer (split view)            │    │
│   │    • /errors     → Traces with potential errors                    │    │
│   │    • /traces     → Paginated list of all traces                    │    │
│   └──────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
TraceAudit-claude/
├── config.py           # Configuration constants (paths, settings)
├── data_processor.py   # ETL pipeline for processing JSONL traces
├── analyzer.py         # Analysis engine with statistics functions
├── components.py       # FastHTML UI components
├── main.py             # FastHTML web server and routes
├── data/
│   └── *.jsonl         # Raw LLM trace data
├── prompts/
│   └── prompt_*.txt    # Extracted system prompts
├── output/
│   ├── traces.db       # SQLite database with processed traces
│   ├── analysis_cache.json
│   └── processed/
└── requirements.txt
```

---

## Detailed Workflow

### Step 1: Data Processing (`data_processor.py`)

**Entry Point:** `python data_processor.py`

This is the ETL (Extract-Transform-Load) pipeline that processes raw JSONL trace data:

1. **Initialize Database**
   - Creates SQLite database (`traces.db`) with three tables:
     - `traces`: Processed trace metadata
     - `prompts`: System prompt tracking
     - `raw_traces`: Full trace JSON for detail view

2. **Parse Each Trace Line**
   - Extract trace metadata (ID, timestamp, model, cost, tokens)
   - Extract messages from `request.messages`
   - Extract response data from `response.choices`

3. **System Prompt Extraction**
   - Find system message in conversation
   - Generate MD5 hash (first 16 chars) as unique identifier
   - Save prompt content to `prompts/prompt_{hash}.txt`

4. **Error Detection**
   - **Multi-turn conversations:** Detect user disagreement phrases ("that's wrong", "incorrect", etc.)
   - **Single-turn interactions:** Check for empty responses, error status codes, refusal patterns

5. **Metadata Extraction**
   - Check for `x-portkey-metadata` and other metadata fields
   - Track which traces have meaningful metadata

6. **Store Results**
   - Insert processed trace info into `traces` table
   - Store raw JSON in `raw_traces` for detail viewing

### Step 2: Analysis (`analyzer.py`)

**Functions:**

| Function | Purpose |
|----------|---------|
| `run_analysis()` | Compute comprehensive statistics, returns `AnalysisResult` dataclass |
| `get_random_traces_for_use_case()` | Get random sample traces for a specific system prompt |
| `get_trace_detail()` | Get full trace details including raw data |
| `get_system_prompt_content()` | Read system prompt from file |
| `get_traces_with_errors()` | Get traces with error indicators |
| `get_model_stats_by_use_case()` | Model usage breakdown per use case |

**Analysis Metrics:**
- Total traces count
- Unique use cases (distinct system prompts)
- Time period and span
- Model distribution
- Multi-turn vs single-turn statistics
- Tool call statistics
- Metadata presence statistics
- Error detection statistics
- Cost statistics (total, avg, min, max)
- Token statistics
- Response time statistics

### Step 3: Web Dashboard (`main.py` + `components.py`)

**Launch:** `python main.py` → Server runs at `http://127.0.0.1:8000`

#### Routes:

| Route | Page | Description |
|-------|------|-------------|
| `/` | Dashboard | Overview stats, model distribution chart, use case preview, error/metadata panels |
| `/use-cases` | Use Cases List | All unique system prompts with trace counts |
| `/use-case/{hash}` | Use Case Detail | Split view: sample traces (left) + system prompt (right) |
| `/trace/{id}` | Trace Detail | Split view: conversation messages (left) + system prompt & metadata (right) |
| `/errors` | Error Traces | Traces flagged with potential errors or user disagreements |
| `/traces` | All Traces | Paginated list of all processed traces |

#### UI Components (`components.py`):

| Component | Purpose |
|-----------|---------|
| `StatCard` | Display single metric with value and optional subtitle |
| `ModelDistributionChart` | Horizontal bar chart for model usage |
| `UseCaseCard` | Clickable card showing use case summary |
| `TraceListItem` | Trace preview with model, turns, cost, error badges |
| `MessageBubble` | Styled message display with role-based colors |
| `ErrorDetectionPanel` | Error summary and breakdown |
| `MetadataStatsPanel` | Metadata presence and key analysis |
| `TokenCostPanel` | Performance metrics display |
| `NavBar` | Navigation between dashboard sections |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Process trace data (runs on TEST_SAMPLE_SIZE=1000 traces by default)
python data_processor.py

# 3. Launch web dashboard
python main.py

# 4. Open browser to http://127.0.0.1:8000
```

---

## Key Features

### Use Case Identification
- System prompts are hashed and deduplicated
- Each unique system prompt = one "use case"
- Traces are grouped by use case for analysis

### Error Detection Methods
1. **Multi-turn:** User disagreement detection via phrase matching
2. **Single-turn:**
   - Non-200 status codes
   - Empty response choices/content
   - Refusal patterns ("I can't", "I cannot", etc.)
   - Response errors

### Split View Interface
Individual traces show:
- **Left Panel:** Full conversation with turn numbers, model badges, tool calls
- **Right Panel:** System prompt content + trace metadata
- System prompts in messages replaced with placeholder references

### Metadata Tracking
Checks multiple fields for metadata presence:
- `metadata` object
- `portkeyHeaders`
- `x-portkey-metadata`

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `TEST_SAMPLE_SIZE` | 1000 | Traces to process in test run |
| `FULL_SAMPLE_SIZE` | 50000 | Full dataset size |
| `HOST` | 127.0.0.1 | Server host |
| `PORT` | 8000 | Server port |

---

## Data Flow Summary

```
JSONL File → data_processor.py → SQLite DB + Prompt Files
                                        ↓
                               analyzer.py (queries)
                                        ↓
                               main.py (renders pages)
                                        ↓
                               Browser (user interaction)
```
