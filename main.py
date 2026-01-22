from fasthtml.common import *
import logging
import json
import subprocess
import argparse
from pathlib import Path

from config import HOST, PORT, PROMPTS_DIR, SAMPLE_SIZE

# Global config that can be set via CLI args
app_config = {
    'sample_size': SAMPLE_SIZE
}
from analyzer import (
    run_analysis, get_random_traces_for_use_case, get_trace_detail,
    get_system_prompt_content, get_traces_with_errors, get_model_stats_by_use_case,
    get_traces_with_multiple_elements, get_filtered_traces, get_available_filters,
    get_element_detail, get_use_case_info
)
from components import (
    StatCard, ModelDistributionChart, UseCaseDistributionChart, UseCaseCard, TraceListItem,
    MessageBubble, ErrorDetectionPanel, MetadataStatsPanel, TokenCostPanel, NavBar,
    MultiElementStatsPanel, FilterPanel, ErrorFilterPanel, format_date
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('traceaudit.log')
    ]
)
logger = logging.getLogger(__name__)

app, rt = fast_app(
    live=False,
    hdrs=(
        Script(src="https://unpkg.com/htmx.org@1.9.10"),
        Style("""
            *, *::before, *::after {
                box-sizing: border-box;
            }
            html, body {
                margin: 0;
                padding: 0;
                height: auto;
                min-height: 100vh;
                overflow-x: hidden;
                overflow-y: scroll !important;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f1a;
                color: #e2e8f0;
                line-height: 1.5;
            }
            .main-container {
                max-width: 1600px;
                margin: 0 auto;
                padding: 24px;
                padding-bottom: 80px;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }
            .use-case-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                gap: 16px;
            }
            .use-case-card {
                cursor: pointer !important;
                transition: transform 0.2s, border-color 0.2s;
            }
            .use-case-card:hover {
                transform: translateY(-2px);
                border-color: #3b82f6 !important;
            }
            .trace-item {
                cursor: pointer !important;
                transition: border-color 0.2s;
            }
            .trace-item:hover {
                border-color: #3b82f6 !important;
            }
            .split-view {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
                min-height: calc(100vh - 200px);
            }
            .split-panel {
                background: #16213e;
                border-radius: 12px;
                border: 1px solid #2d3748;
                overflow-y: auto;
                padding: 20px;
                max-height: calc(100vh - 200px);
            }
            .section-header {
                margin-bottom: 20px;
            }
            .section-header h2 {
                margin: 0 0 8px 0;
                color: #e2e8f0;
            }
            .section-header p {
                margin: 0;
                color: #94a3b8;
                font-size: 0.9em;
            }
            a {
                color: #3b82f6;
                text-decoration: none;
                cursor: pointer !important;
            }
            a:hover {
                color: #60a5fa;
            }
            pre {
                background: #0f0f1a;
                padding: 16px;
                border-radius: 8px;
                overflow-x: auto;
                font-size: 0.85em;
                line-height: 1.5;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .back-link {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                color: #94a3b8;
                text-decoration: none;
                margin-bottom: 16px;
                cursor: pointer !important;
            }
            .back-link:hover {
                color: #e2e8f0;
            }
            button {
                cursor: pointer !important;
                border: none;
                outline: none;
            }
            button:hover {
                opacity: 0.9;
            }
            button:active {
                transform: scale(0.98);
            }
            .page-btn {
                display: inline-block;
                padding: 10px 20px;
                border-radius: 6px;
                text-decoration: none;
                cursor: pointer !important;
                font-weight: 500;
                transition: background 0.2s;
            }
            .page-btn-active {
                background: #3b82f6;
                color: white;
            }
            .page-btn-active:hover {
                background: #2563eb;
                color: white;
            }
            .page-btn-disabled {
                background: #2d3748;
                color: #64748b;
                pointer-events: none;
                cursor: default !important;
            }
            h1, h2, h3, h4, h5, h6 {
                margin: 0;
                color: #e2e8f0;
            }
            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #1a1a2e;
            }
            ::-webkit-scrollbar-thumb {
                background: #3b82f6;
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #2563eb;
            }
        """)
    )
)


@rt('/reprocess')
def post():
    """Rerun the data processor to reprocess all traces"""
    import time as time_module
    start_time = time_module.time()

    sample_size = app_config['sample_size']

    logger.info("=" * 60)
    logger.info("REPROCESS REQUEST RECEIVED")
    logger.info("=" * 60)
    logger.info(f"Sample size: {sample_size:,}")
    logger.info("Launching data processor subprocess...")

    try:
        result = subprocess.run(
            ['python', 'data_processor.py', '--size', str(sample_size)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent)
        )

        elapsed = time_module.time() - start_time

        # Log stdout (the processing output)
        if result.stdout:
            logger.info("-" * 40)
            logger.info("DATA PROCESSOR OUTPUT:")
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  {line}")

        # Log stderr (usually contains the detailed logging)
        if result.stderr:
            logger.info("-" * 40)
            logger.info("DATA PROCESSOR LOGS:")
            for line in result.stderr.strip().split('\n')[-20:]:  # Last 20 lines
                logger.info(f"  {line}")

        if result.returncode == 0:
            logger.info("-" * 40)
            logger.info(f"REPROCESS COMPLETED SUCCESSFULLY in {elapsed:.2f}s")
            logger.info("=" * 60)
            # Return a redirect response
            return RedirectResponse('/', status_code=303)
        else:
            logger.error("-" * 40)
            logger.error(f"REPROCESS FAILED (exit code: {result.returncode})")
            logger.error(f"Error output: {result.stderr[-500:] if result.stderr else 'No error output'}")
            logger.error("=" * 60)
            return Div(
                H3("Reprocess Failed", style="color: #ef4444; margin-bottom: 16px;"),
                Pre(result.stderr or "Unknown error", style="background: #1f2940; padding: 16px; border-radius: 8px; color: #f87171; max-height: 400px; overflow-y: auto;"),
                A("Back to Dashboard", href="/", style="display: inline-block; margin-top: 16px; padding: 12px 24px; background: #3b82f6; color: white; border-radius: 8px; text-decoration: none;"),
                style="padding: 24px;"
            )
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"REPROCESS EXCEPTION: {str(e)}")
        logger.error("=" * 60)
        return Div(
            H3("Reprocess Error", style="color: #ef4444; margin-bottom: 16px;"),
            P(f"Exception: {str(e)}", style="color: #f87171;"),
            A("Back to Dashboard", href="/", style="display: inline-block; margin-top: 16px; padding: 12px 24px; background: #3b82f6; color: white; border-radius: 8px; text-decoration: none;"),
            style="padding: 24px;"
        )


@rt('/')
def get():
    logger.info("Loading dashboard")
    try:
        analysis = run_analysis()
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return Div(
            NavBar("dashboard"),
            Div(
                H2("No Data Available"),
                P("Please run the data processor first: python data_processor.py"),
                cls="main-container"
            )
        )

    # Format time period for display with highlighting
    time_start = format_date(analysis.time_period_start) if analysis.time_period_start != 'Unknown' else 'Unknown'
    time_end = format_date(analysis.time_period_end) if analysis.time_period_end != 'Unknown' else 'Unknown'
    time_span = f"{analysis.time_span_days:.0f}" if analysis.time_span_days else "N/A"

    return Div(
        NavBar("dashboard"),
        Div(
            Div(
                Div(
                    Div(
                        H2("📊 TraceAudit Dashboard", style="margin: 0;"),
                        P(
                            Span(f"Analyzing {analysis.total_elements:,} traces ({analysis.unique_trace_ids:,} unique) from "),
                            Span(time_start, style="color: #3b82f6; font-weight: 600;"),
                            Span(" to "),
                            Span(time_end, style="color: #3b82f6; font-weight: 600;"),
                            Span(" · "),
                            Span(f"{time_span} days", style="color: #10b981; font-weight: 600;"),
                            style="margin: 4px 0 0 0; color: #94a3b8;"
                        ),
                    ),
                    Form(
                        Button(
                            "🔄 Rerun Process",
                            type="submit",
                            style="padding: 10px 20px; background: #8b5cf6; color: white; border: none; border-radius: 8px; font-weight: 500; cursor: pointer;"
                        ),
                        action="/reprocess",
                        method="post"
                    ),
                    style="display: flex; justify-content: space-between; align-items: flex-start;"
                ),
                cls="section-header"
            ),
            # Top stats - Unique traces first, total traces last
            Div(
                StatCard("Unique Traces", f"{analysis.unique_trace_ids:,}", color="#3b82f6"),
                StatCard("Unique Use Cases", analysis.unique_use_cases, color="#10b981"),
                StatCard("Multi-turn", analysis.multi_turn_stats['multi_turn_count'], f"{analysis.multi_turn_stats['avg_turns']:.1f} avg turns", "#8b5cf6"),
                StatCard("With Metadata", analysis.metadata_stats.get('with_metadata', 0), color="#f59e0b"),
                StatCard("Total Traces", f"{analysis.total_elements:,}", color="#6b7280"),
                cls="stats-grid"
            ),
            # Model and Use Case distribution pie charts side by side
            Div(
                ModelDistributionChart(analysis.model_distribution),
                UseCaseDistributionChart(analysis.use_case_distribution),
                style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px; background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748;"
            ),
            # Two column layout for error and metadata
            Div(
                ErrorDetectionPanel(analysis.error_detection_stats),
                MetadataStatsPanel(analysis.metadata_stats),
                style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px;"
            ),
            # Use cases preview
            Div(
                H3("Use Cases", style="margin-bottom: 16px; color: #e2e8f0;"),
                Div(
                    *[UseCaseCard(hash, info, i) for i, (hash, info) in enumerate(list(analysis.use_case_distribution.items())[:6])],
                    cls="use-case-grid"
                ),
                A("View All Use Cases →", href="/use-cases", style="display: block; margin-top: 16px; text-align: right;"),
                style="margin-top: 24px;"
            ),
            cls="main-container"
        )
    )


@rt('/use-cases')
def get(type_filter: str = None):
    logger.info(f"Loading use cases page with type_filter={type_filter}")
    analysis = run_analysis()

    # Get all unique types for filter dropdown and count matched/unmatched
    all_types = set()
    unmatched_count = 0
    matched_count = 0
    for info in analysis.use_case_distribution.values():
        if info.get('use_case_type'):
            # Split comma-separated types and add each individually
            for t in info['use_case_type'].split(', '):
                if t.strip():
                    all_types.add(t.strip())
        if not info.get('use_case_name'):
            unmatched_count += 1
        else:
            matched_count += 1
    all_types = sorted(all_types)

    # Filter use cases by type if filter is set
    filtered_use_cases = {}
    for hash_val, info in analysis.use_case_distribution.items():
        if type_filter == 'unmatched':
            # Show only unmatched (no use_case_name)
            if not info.get('use_case_name'):
                filtered_use_cases[hash_val] = info
        elif type_filter == 'matched':
            # Show only matched (has use_case_name)
            if info.get('use_case_name'):
                filtered_use_cases[hash_val] = info
        elif type_filter and type_filter != 'all':
            # Filter by type - check if the filter type is in the comma-separated list
            use_case_types = [t.strip() for t in (info.get('use_case_type') or '').split(', ') if t.strip()]
            if type_filter in use_case_types:
                filtered_use_cases[hash_val] = info
        else:
            filtered_use_cases[hash_val] = info

    # Type filter buttons
    type_colors = {'workflows': '#8b5cf6', 'agents': '#10b981', 'prompts': '#f59e0b'}
    type_buttons = [
        A(
            "All",
            href="/use-cases",
            style=f"padding: 8px 16px; border-radius: 6px; text-decoration: none; color: {'white' if not type_filter or type_filter == 'all' else '#94a3b8'}; background: {'#3b82f6' if not type_filter or type_filter == 'all' else '#1f2940'}; font-weight: {'600' if not type_filter or type_filter == 'all' else '400'};"
        )
    ]
    for t in all_types:
        color = type_colors.get(t, '#6b7280')
        is_active = type_filter == t
        type_buttons.append(
            A(
                t.capitalize(),
                href=f"/use-cases?type_filter={t}",
                style=f"padding: 8px 16px; border-radius: 6px; text-decoration: none; color: {'white' if is_active else '#94a3b8'}; background: {color if is_active else '#1f2940'}; font-weight: {'600' if is_active else '400'};"
            )
        )
    # Add Matched filter button
    is_matched_active = type_filter == 'matched'
    type_buttons.append(
        A(
            f"Matched ({matched_count})",
            href="/use-cases?type_filter=matched",
            style=f"padding: 8px 16px; border-radius: 6px; text-decoration: none; color: {'white' if is_matched_active else '#94a3b8'}; background: {'#22c55e' if is_matched_active else '#1f2940'}; font-weight: {'600' if is_matched_active else '400'};"
        )
    )
    # Add Unmatched filter button
    is_unmatched_active = type_filter == 'unmatched'
    type_buttons.append(
        A(
            f"Unmatched ({unmatched_count})",
            href="/use-cases?type_filter=unmatched",
            style=f"padding: 8px 16px; border-radius: 6px; text-decoration: none; color: {'white' if is_unmatched_active else '#94a3b8'}; background: {'#dc2626' if is_unmatched_active else '#1f2940'}; font-weight: {'600' if is_unmatched_active else '400'};"
        )
    )

    # Build filter description
    filter_desc = ""
    if type_filter == 'unmatched':
        filter_desc = " (unmatched use cases)"
    elif type_filter and type_filter != 'all':
        filter_desc = f" (filtered by: {type_filter})"

    return Div(
        NavBar("use-cases"),
        Div(
            A("← Back to Dashboard", href="/", cls="back-link"),
            Div(
                H2("🎯 Use Cases", style="margin: 0;"),
                P(f"{len(filtered_use_cases)} of {analysis.unique_use_cases} use cases{filter_desc}", style="margin: 4px 0 0 0; color: #94a3b8;"),
                cls="section-header"
            ),
            # Type filter
            Div(
                Span("Filter by type:", style="color: #94a3b8; margin-right: 12px;"),
                *type_buttons,
                style="display: flex; align-items: center; gap: 8px; margin-bottom: 24px; flex-wrap: wrap;"
            ),
            Div(
                *[UseCaseCard(hash, info, i) for i, (hash, info) in enumerate(filtered_use_cases.items())],
                cls="use-case-grid"
            ) if filtered_use_cases else Div(
                P("No use cases found for this filter", style="color: #64748b; text-align: center; padding: 40px;")
            ),
            cls="main-container"
        )
    )


@rt('/use-case/{prompt_hash}')
def get(prompt_hash: str):
    logger.info(f"Loading use case: {prompt_hash}")

    # Get random traces for this use case
    traces = get_random_traces_for_use_case(prompt_hash, limit=10)
    system_prompt = get_system_prompt_content(prompt_hash)

    # Get model stats for this use case
    model_stats = get_model_stats_by_use_case().get(prompt_hash, [])

    # Get use case info (name, type, workspace, version)
    use_case_info = get_use_case_info(prompt_hash)
    use_case_name = use_case_info.get('use_case_name') if use_case_info else None
    use_case_type = use_case_info.get('use_case_type') if use_case_info else None
    use_case_version = use_case_info.get('use_case_version') if use_case_info else None
    workspace = use_case_info.get('workspace') if use_case_info else None

    # Build title
    title = use_case_name if use_case_name else f"Use Case: {prompt_hash[:8]}..."

    # Type badges - create separate badge for each type
    type_badges = []
    type_colors = {'workflows': '#8b5cf6', 'agents': '#10b981', 'prompts': '#f59e0b'}
    if use_case_type:
        for t in use_case_type.split(', '):
            t = t.strip()
            if t:
                color = type_colors.get(t, '#6b7280')
                type_badges.append(
                    Span(t, style=f"background: {color}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 0.8em;")
                )

    # Version badge
    version_badge = None
    if use_case_version:
        version_badge = Span(use_case_version, style="background: #374151; color: #d1d5db; padding: 4px 12px; border-radius: 4px; font-size: 0.8em;")

    return Div(
        NavBar("use-cases"),
        Div(
            A("← Back to Use Cases", href="/use-cases", cls="back-link"),
            Div(
                Div(
                    H2(title, style="margin: 0; display: inline;"),
                    *type_badges,
                    version_badge,
                    style="display: flex; align-items: center; flex-wrap: wrap; gap: 8px;"
                ),
                P(
                    Span(f"{len(traces)} sample traces loaded"),
                    Span(f" · Workspace: {workspace}", style="color: #64748b;") if workspace else None,
                    style="margin: 4px 0 0 0; color: #94a3b8;"
                ),
                cls="section-header"
            ),
            Div(
                # Left panel - Trace list
                Div(
                    H3("Sample Traces", style="margin-bottom: 16px; color: #e2e8f0;"),
                    # Model stats for this use case
                    Div(
                        H4("Models Used", style="margin-bottom: 12px; color: #94a3b8; font-size: 0.85em;"),
                        *[
                            Div(
                                Span(stat['model'].split('/')[-1], style="flex: 1; color: #e2e8f0;"),
                                Span(f"{stat['count']} traces", style="color: #3b82f6;"),
                                style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #2d3748;"
                            )
                            for stat in model_stats
                        ],
                        style="background: #0f0f1a; padding: 12px; border-radius: 8px; margin-bottom: 16px;"
                    ) if model_stats else None,
                    *[TraceListItem(trace) for trace in traces],
                    Button("Load More Random Traces",
                           hx_get=f"/use-case/{prompt_hash}/refresh",
                           hx_target="#trace-list",
                           hx_swap="innerHTML",
                           style="margin-top: 16px; padding: 12px 24px; background: #3b82f6; color: white; border: none; border-radius: 8px; cursor: pointer; width: 100%;"),
                    id="trace-list",
                    cls="split-panel"
                ),
                # Right panel - System prompt
                Div(
                    H3("System Prompt", style="margin-bottom: 16px; color: #e2e8f0;"),
                    Div(
                        Span(f"File: prompt_{prompt_hash}.txt", style="font-size: 0.85em; color: #64748b; font-family: monospace;"),
                        style="margin-bottom: 12px;"
                    ),
                    Pre(system_prompt, style="white-space: pre-wrap; color: #e2e8f0; max-height: calc(100vh - 300px); overflow-y: auto;"),
                    cls="split-panel"
                ),
                cls="split-view"
            ),
            cls="main-container"
        )
    )


@rt('/use-case/{prompt_hash}/refresh')
def get(prompt_hash: str):
    traces = get_random_traces_for_use_case(prompt_hash, limit=10)
    return Div(
        *[TraceListItem(trace) for trace in traces],
        Button("Load More Random Traces",
               hx_get=f"/use-case/{prompt_hash}/refresh",
               hx_target="#trace-list",
               hx_swap="innerHTML",
               style="margin-top: 16px; padding: 12px 24px; background: #3b82f6; color: white; border: none; border-radius: 8px; cursor: pointer; width: 100%;"),
    )


@rt('/trace/{trace_id}')
def get(trace_id: str, element_idx: int = 0):
    logger.info(f"Loading trace: {trace_id}, element: {element_idx}")

    trace_detail = get_trace_detail(trace_id)
    if not trace_detail:
        return Div(
            NavBar("traces"),
            Div(H2("Trace not found"), cls="main-container")
        )

    elements = trace_detail.get('elements', [])
    element_count = len(elements)

    # Get the selected element (default to first)
    if element_idx >= element_count:
        element_idx = 0
    current_element = elements[element_idx] if elements else {}

    raw_data = current_element.get('raw_data', {})
    request = raw_data.get('request', {})
    response = raw_data.get('response', {})
    messages = request.get('messages', []).copy()

    # Add response message to messages list
    choices = response.get('choices', [])
    if choices:
        response_msg = choices[0].get('message', {})
        if response_msg:
            messages.append(response_msg)

    # Get model
    model = current_element.get('model', raw_data.get('ai_model', 'unknown'))

    # Get system prompt
    system_prompt = get_system_prompt_content(trace_detail.get('system_prompt_hash', ''))

    # Get use case info
    use_case_info = get_use_case_info(trace_detail.get('system_prompt_hash', ''))
    use_case_name = use_case_info.get('use_case_name') if use_case_info else None

    # Calculate turn numbers
    turn_numbers = {}
    current_turn = 0
    for idx, msg in enumerate(messages):
        if msg.get('role') == 'user':
            current_turn += 1
            turn_numbers[idx] = current_turn
        elif msg.get('role') == 'assistant':
            turn_numbers[idx] = current_turn

    # Render messages
    rendered_messages = []
    for idx, msg in enumerate(messages):
        rendered_messages.append(
            MessageBubble(msg, idx, model=model if msg.get('role') == 'assistant' else None,
                         turn_number=turn_numbers.get(idx))
        )

    # Trace selector if multiple traces
    element_selector = None
    if element_count > 1:
        element_tabs = []
        for i, el in enumerate(elements):
            is_active = i == element_idx
            element_tabs.append(
                A(
                    f"Trace {i + 1}",
                    Span(format_date(el.get('created_at', '')), style="display: block; font-size: 0.7em; color: #64748b;"),
                    href=f"/trace/{trace_id}?element_idx={i}",
                    style=f"padding: 12px 16px; background: {'#3b82f6' if is_active else '#1f2940'}; color: {'white' if is_active else '#94a3b8'}; border-radius: 8px; text-decoration: none; text-align: center; min-width: 100px;"
                )
            )
        element_selector = Div(
            Div(
                Span(f"📦 {element_count} Traces", style="font-weight: 600; color: #8b5cf6; margin-right: 16px;"),
                Span("This trace ID has multiple traces - click to view each one", style="color: #64748b; font-size: 0.85em;"),
                style="margin-bottom: 12px;"
            ),
            Div(*element_tabs, style="display: flex; gap: 8px; flex-wrap: wrap;"),
            style="background: #16213e; padding: 16px; border-radius: 12px; border: 1px solid #8b5cf6; margin-bottom: 16px;"
        )

    return Div(
        NavBar("traces"),
        Div(
            A("← Back", href="javascript:history.back()", cls="back-link"),
            Div(
                H2(f"Trace: {trace_id[:16]}...", style="margin: 0;"),
                Div(
                    A(use_case_name, href=f"/use-case/{trace_detail.get('system_prompt_hash', '')}",
                      style="background: #10b981; color: white; padding: 4px 12px; border-radius: 4px; font-size: 0.85em; margin-right: 12px; text-decoration: none;") if use_case_name else Span(f"Use Case: {trace_detail.get('system_prompt_hash', '')[:8]}...", style="background: #6b7280; color: white; padding: 4px 12px; border-radius: 4px; font-size: 0.85em; margin-right: 12px;"),
                    Span(f"Model: {model}", style="background: #3b82f6; color: white; padding: 4px 12px; border-radius: 4px; font-size: 0.85em; margin-right: 12px;"),
                    Span(f"{current_element.get('num_turns', 0)} turns", style="color: #94a3b8; margin-right: 12px;"),
                    Span(f"${current_element.get('cost', 0):.4f}", style="color: #10b981; margin-right: 12px;"),
                    Span(f"{current_element.get('total_tokens', 0):,} tokens", style="color: #8b5cf6;"),
                    Span(f"📦 Trace {element_idx + 1}/{element_count}", style="margin-left: 12px; color: #8b5cf6;") if element_count > 1 else None,
                    style="margin-top: 8px;"
                ),
                cls="section-header"
            ),
            element_selector,
            Div(
                # Left panel - Messages
                Div(
                    H3("Conversation", style="margin-bottom: 16px; color: #e2e8f0;"),
                    Div(f"Messages: {len(messages)}", style="font-size: 0.85em; color: #64748b; margin-bottom: 12px;"),
                    *rendered_messages,
                    cls="split-panel"
                ),
                # Right panel - System Prompt
                Div(
                    H3("System Prompt", style="margin-bottom: 16px; color: #e2e8f0;"),
                    Div(
                        Span(f"Placeholder: prompt_{trace_detail.get('system_prompt_hash', '')}.txt",
                             style="font-size: 0.85em; color: #64748b; font-family: monospace;"),
                        style="margin-bottom: 12px;"
                    ),
                    # Trace metadata
                    Div(
                        H4("Trace Metadata", style="margin-bottom: 12px; color: #94a3b8; font-size: 0.85em;"),
                        Div(f"Use Case: {use_case_name or 'Unknown'}", style="font-size: 0.85em; color: #10b981; margin-bottom: 4px; font-weight: 600;"),
                        Div(f"Trace ID: {current_element.get('element_id', 'N/A')[:16]}...", style="font-size: 0.85em; color: #64748b; margin-bottom: 4px;"),
                        Div(f"Created: {format_date(current_element.get('created_at', 'N/A'))}", style="font-size: 0.85em; color: #64748b; margin-bottom: 4px;"),
                        Div(f"Response Time: {current_element.get('response_time', 0)}ms", style="font-size: 0.85em; color: #64748b; margin-bottom: 4px;"),
                        Div(f"Has Tool Calls: {'Yes' if current_element.get('has_tool_calls') else 'No'}", style="font-size: 0.85em; color: #64748b; margin-bottom: 4px;"),
                        Div(f"Error Indicators: {current_element.get('potential_error_indicators') or 'None'}", style=f"font-size: 0.85em; color: {'#f87171' if current_element.get('potential_error_indicators') else '#64748b'}; margin-bottom: 4px;"),
                        style="background: #0f0f1a; padding: 12px; border-radius: 8px; margin-bottom: 16px;"
                    ),
                    Pre(system_prompt, style="white-space: pre-wrap; color: #e2e8f0; max-height: calc(100vh - 450px); overflow-y: auto;"),
                    cls="split-panel"
                ),
                cls="split-view"
            ),
            Script("""
                function toggleSystemMsg(contentId, toggleId) {
                    const content = document.getElementById(contentId);
                    const toggle = document.getElementById(toggleId);
                    if (content && toggle) {
                        if (content.style.display === 'none') {
                            content.style.display = 'block';
                            toggle.textContent = '▼';
                        } else {
                            content.style.display = 'none';
                            toggle.textContent = '▶';
                        }
                    }
                }
            """),
            cls="main-container"
        )
    )


@rt('/errors')
def get(model: str = None, use_case: str = None, error_type: str = None):
    logger.info(f"Loading errors page with filters: model={model}, use_case={use_case}, error_type={error_type}")

    traces = get_traces_with_errors(limit=100, model=model, use_case=use_case, error_type=error_type)
    analysis = run_analysis()
    filters = get_available_filters()

    return Div(
        NavBar("errors"),
        Div(
            A("← Back to Dashboard", href="/", cls="back-link"),
            Div(
                H2("⚠️ Traces with Potential Errors", style="margin: 0;"),
                P(f"{len(traces)} traces found with error indicators or user disagreements", style="margin: 4px 0 0 0; color: #94a3b8;"),
                cls="section-header"
            ),
            # Error detection explanation
            Div(
                H3("Error Detection Methods", style="margin-bottom: 12px; color: #e2e8f0;"),
                Div(
                    Div(
                        H4("Multi-turn Conversations", style="margin: 0 0 8px 0; color: #f59e0b;"),
                        P("Detects when users express disagreement with LLM responses through phrases like 'that's wrong', 'incorrect', 'not what I asked', etc.",
                          style="margin: 0; color: #94a3b8; font-size: 0.9em;"),
                        style="flex: 1; padding: 16px; background: #2d2d1f; border-radius: 8px; border-left: 4px solid #f59e0b;"
                    ),
                    Div(
                        H4("Single-turn Interactions", style="margin: 0 0 8px 0; color: #ef4444;"),
                        P("Checks for: empty responses, error status codes, refusal patterns ('I can't', 'I cannot'), and response errors.",
                          style="margin: 0; color: #94a3b8; font-size: 0.9em;"),
                        style="flex: 1; padding: 16px; background: #2d1f1f; border-radius: 8px; border-left: 4px solid #ef4444;"
                    ),
                    style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;"
                ),
                style="background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748; margin-bottom: 24px;"
            ),
            # Error stats
            ErrorDetectionPanel(analysis.error_detection_stats),
            # Filter panel and trace list side by side
            Div(
                # Filter panel on the left
                Div(
                    ErrorFilterPanel(filters, current_model=model, current_use_case=use_case, error_type=error_type),
                    style="width: 280px; flex-shrink: 0;"
                ),
                # Trace list on the right
                Div(
                    H3("Flagged Traces", style="margin-bottom: 16px; color: #e2e8f0;"),
                    *[TraceListItem(trace, show_use_case=True) for trace in traces] if traces else [
                        Div(P("No traces with errors found", style="color: #64748b; text-align: center; padding: 40px;"))
                    ],
                    style="flex: 1; max-height: 600px; overflow-y: auto;"
                ),
                style="display: flex; gap: 24px; margin-top: 24px;"
            ),
            cls="main-container"
        )
    )


@rt('/traces')
def get(page: int = 1, limit: int = 50, model: str = None, use_case: str = None, has_errors: str = None, multi_element_only: str = None, with_metadata: str = None, multi_turn: str = None):
    logger.info(f"Loading all traces page {page} with filters")

    # Parse boolean filters
    has_errors_bool = has_errors == 'true' if has_errors else None
    multi_element_bool = multi_element_only == 'true' if multi_element_only else False
    with_metadata_bool = with_metadata == 'true' if with_metadata else None
    multi_turn_bool = multi_turn == 'true' if multi_turn else None

    # Get filtered traces
    traces, total = get_filtered_traces(
        page=page,
        limit=limit,
        model=model if model else None,
        use_case=use_case if use_case else None,
        has_errors=has_errors_bool,
        multi_element_only=multi_element_bool,
        with_metadata=with_metadata_bool,
        multi_turn=multi_turn_bool
    )

    # Get available filters
    filters = get_available_filters()

    total_pages = (total + limit - 1) // limit
    offset = (page - 1) * limit

    # Build query string for pagination
    query_parts = []
    if model:
        query_parts.append(f"model={model}")
    if use_case:
        query_parts.append(f"use_case={use_case}")
    if has_errors:
        query_parts.append(f"has_errors={has_errors}")
    if multi_element_only:
        query_parts.append(f"multi_element_only={multi_element_only}")
    if with_metadata:
        query_parts.append(f"with_metadata={with_metadata}")
    if multi_turn:
        query_parts.append(f"multi_turn={multi_turn}")
    query_string = "&".join(query_parts)
    query_suffix = f"&{query_string}" if query_string else ""

    return Div(
        NavBar("traces"),
        Div(
            A("← Back to Dashboard", href="/", cls="back-link"),
            Div(
                H2("📋 All Traces", style="margin: 0;"),
                P(f"Showing {offset + 1}-{min(offset + limit, total)} of {total:,} traces", style="margin: 4px 0 0 0; color: #94a3b8;"),
                cls="section-header"
            ),
            Div(
                # Filter panel on the left
                Div(
                    FilterPanel(filters, current_model=model, current_use_case=use_case, has_errors=has_errors_bool, multi_element_only=multi_element_bool, with_metadata=with_metadata_bool, multi_turn=multi_turn_bool),
                    style="width: 280px; flex-shrink: 0;"
                ),
                # Traces list on the right
                Div(
                    *[TraceListItem(trace) for trace in traces],
                    # Pagination
                    Div(
                        A("← Previous",
                          href=f"/traces?page={page-1}{query_suffix}" if page > 1 else None,
                          cls=f"page-btn {'page-btn-active' if page > 1 else 'page-btn-disabled'}",
                          style="margin-right: 12px;"),
                        Span(f"Page {page} of {total_pages}", style="color: #94a3b8; font-weight: 500;"),
                        A("Next →",
                          href=f"/traces?page={page+1}{query_suffix}" if page < total_pages else None,
                          cls=f"page-btn {'page-btn-active' if page < total_pages else 'page-btn-disabled'}",
                          style="margin-left: 12px;"),
                        style="display: flex; align-items: center; justify-content: center; margin-top: 24px; padding: 16px;"
                    ),
                    style="flex: 1;"
                ),
                style="display: flex; gap: 24px;"
            ),
            cls="main-container"
        )
    )


@rt('/multi-element')
def get():
    logger.info("Loading multi-trace groups page")

    traces = get_traces_with_multiple_elements(limit=100)
    analysis = run_analysis()

    return Div(
        NavBar("multi-element"),
        Div(
            A("← Back to Dashboard", href="/", cls="back-link"),
            Div(
                H2("📦 Trace IDs with Multiple Traces", style="margin: 0;"),
                P(f"{len(traces)} trace IDs found with multiple traces", style="margin: 4px 0 0 0; color: #94a3b8;"),
                cls="section-header"
            ),
            # Stats panel
            MultiElementStatsPanel(analysis.multi_element_stats),
            # Explanation
            Div(
                H3("What are Multi-Trace Groups?", style="margin-bottom: 12px; color: #e2e8f0;"),
                P("Each trace ID can have multiple traces representing different stages or retries of the same conversation. "
                  "These might occur due to retries, different model calls, or progressive conversation building.",
                  style="color: #94a3b8; font-size: 0.9em; line-height: 1.6;"),
                style="background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748; margin: 24px 0;"
            ),
            # Trace list
            Div(
                H3("Multi-Trace Groups", style="margin-bottom: 16px; color: #e2e8f0;"),
                *[
                    A(
                        Div(
                            Div(
                                Div(
                                    Span(f"Trace ID: {trace['trace_id'][:16]}...", style="font-weight: 600; color: #e2e8f0;"),
                                    Span(f"📦 {trace['element_count']} traces", style="background: #8b5cf6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; margin-left: 12px;"),
                                    style="display: flex; align-items: center;"
                                ),
                                Div(
                                    Span(f"Models: {trace['models']}", style="color: #94a3b8; font-size: 0.85em;"),
                                    style="margin-top: 4px;"
                                ),
                                style="flex: 1;"
                            ),
                            Div(
                                Div(f"First: {format_date(trace['first_created'])}", style="font-size: 0.8em; color: #64748b;"),
                                Div(f"Last: {format_date(trace['last_created'])}", style="font-size: 0.8em; color: #64748b;"),
                                style="text-align: right;"
                            ),
                            style="display: flex; justify-content: space-between; padding: 16px;"
                        ),
                        href=f"/trace/{trace['trace_id']}",
                        style="display: block; background: #1f2940; border-radius: 8px; border: 1px solid #2d3748; text-decoration: none; margin-bottom: 8px; transition: all 0.2s;",
                        cls="trace-item"
                    )
                    for trace in traces
                ],
                style="max-height: 600px; overflow-y: auto;"
            ) if traces else Div(
                P("No trace IDs with multiple traces found", style="color: #64748b; text-align: center; padding: 40px;")
            ),
            cls="main-container"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TraceAudit - LLM Trace Analysis Dashboard")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to run the server on (default: {PORT})")
    parser.add_argument("--host", type=str, default=HOST, help=f"Host to bind the server to (default: {HOST})")
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE, help=f"Sample size for processing (default: {SAMPLE_SIZE} from config.py)")
    args = parser.parse_args()

    # Update global config with CLI args
    app_config['sample_size'] = args.size

    logger.info("=" * 60)
    logger.info("TRACEAUDIT SERVER STARTING")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Sample size: {args.size:,}")
    logger.info(f"URL: http://{args.host}:{args.port}")
    logger.info("=" * 60)

    serve(host=args.host, port=args.port)
