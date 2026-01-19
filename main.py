from fasthtml.common import *
import logging
import json
from pathlib import Path

from config import HOST, PORT, PROMPTS_DIR
from analyzer import (
    run_analysis, get_random_traces_for_use_case, get_trace_detail,
    get_system_prompt_content, get_traces_with_errors, get_model_stats_by_use_case
)
from components import (
    StatCard, ModelDistributionChart, UseCaseCard, TraceListItem,
    MessageBubble, ErrorDetectionPanel, MetadataStatsPanel, TokenCostPanel, NavBar
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

    return Div(
        NavBar("dashboard"),
        Div(
            Div(
                H2("📊 TraceAudit Dashboard", style="margin: 0;"),
                P(f"Analyzing {analysis.total_traces:,} traces from {analysis.time_period_start[:10] if analysis.time_period_start != 'Unknown' else 'Unknown'} to {analysis.time_period_end[:10] if analysis.time_period_end != 'Unknown' else 'Unknown'}",
                  style="margin: 4px 0 0 0; color: #94a3b8;"),
                cls="section-header"
            ),
            # Top stats
            Div(
                StatCard("Total Traces", f"{analysis.total_traces:,}", color="#3b82f6"),
                StatCard("Unique Use Cases", analysis.unique_use_cases, color="#10b981"),
                StatCard("Multi-turn", analysis.multi_turn_stats['multi_turn_count'], f"{analysis.multi_turn_stats['avg_turns']:.1f} avg turns", "#8b5cf6"),
                StatCard("With Tool Calls", analysis.tool_call_stats['with_tool_calls'], color="#f59e0b"),
                StatCard("Time Span", f"{analysis.time_span_days:.1f} days" if analysis.time_span_days else "N/A", color="#ec4899"),
                cls="stats-grid"
            ),
            # Model distribution
            ModelDistributionChart(analysis.model_distribution),
            # Token/Cost metrics
            Div(style="margin-top: 24px;"),
            TokenCostPanel(analysis.token_stats, analysis.cost_stats, analysis.response_time_stats),
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
def get():
    logger.info("Loading use cases page")
    analysis = run_analysis()

    return Div(
        NavBar("use-cases"),
        Div(
            A("← Back to Dashboard", href="/", cls="back-link"),
            Div(
                H2("🎯 Use Cases", style="margin: 0;"),
                P(f"{analysis.unique_use_cases} unique system prompts identified", style="margin: 4px 0 0 0; color: #94a3b8;"),
                cls="section-header"
            ),
            Div(
                *[UseCaseCard(hash, info, i) for i, (hash, info) in enumerate(analysis.use_case_distribution.items())],
                cls="use-case-grid"
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

    return Div(
        NavBar("use-cases"),
        Div(
            A("← Back to Use Cases", href="/use-cases", cls="back-link"),
            Div(
                H2(f"Use Case: {prompt_hash[:8]}...", style="margin: 0;"),
                P(f"{len(traces)} sample traces loaded", style="margin: 4px 0 0 0; color: #94a3b8;"),
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
def get(trace_id: str):
    logger.info(f"Loading trace: {trace_id}")

    trace_detail = get_trace_detail(trace_id)
    if not trace_detail:
        return Div(
            NavBar("traces"),
            Div(H2("Trace not found"), cls="main-container")
        )

    raw_data = trace_detail.get('raw_data', {})
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
    model = trace_detail.get('model', raw_data.get('ai_model', 'unknown'))

    # Get system prompt
    system_prompt = get_system_prompt_content(trace_detail.get('system_prompt_hash', ''))

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

    return Div(
        NavBar("traces"),
        Div(
            A("← Back", href="javascript:history.back()", cls="back-link"),
            Div(
                H2(f"Trace: {trace_id[:16]}...", style="margin: 0;"),
                Div(
                    Span(f"Model: {model}", style="background: #3b82f6; color: white; padding: 4px 12px; border-radius: 4px; font-size: 0.85em; margin-right: 12px;"),
                    Span(f"{trace_detail.get('num_turns', 0)} turns", style="color: #94a3b8; margin-right: 12px;"),
                    Span(f"${trace_detail.get('cost', 0):.4f}", style="color: #10b981; margin-right: 12px;"),
                    Span(f"{trace_detail.get('total_tokens', 0):,} tokens", style="color: #8b5cf6;"),
                    style="margin-top: 8px;"
                ),
                cls="section-header"
            ),
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
                        Div(f"Created: {trace_detail.get('created_at', 'N/A')}", style="font-size: 0.85em; color: #64748b; margin-bottom: 4px;"),
                        Div(f"Response Time: {trace_detail.get('response_time', 0)}ms", style="font-size: 0.85em; color: #64748b; margin-bottom: 4px;"),
                        Div(f"Has Tool Calls: {'Yes' if trace_detail.get('has_tool_calls') else 'No'}", style="font-size: 0.85em; color: #64748b; margin-bottom: 4px;"),
                        Div(f"Error Indicators: {trace_detail.get('potential_error_indicators') or 'None'}", style=f"font-size: 0.85em; color: {'#f87171' if trace_detail.get('potential_error_indicators') else '#64748b'}; margin-bottom: 4px;"),
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
def get():
    logger.info("Loading errors page")

    traces = get_traces_with_errors(limit=100)
    analysis = run_analysis()

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
            # Trace list
            Div(
                H3("Flagged Traces", style="margin: 24px 0 16px 0; color: #e2e8f0;"),
                *[TraceListItem(trace, show_use_case=True) for trace in traces],
                style="max-height: 600px; overflow-y: auto;"
            ) if traces else Div(
                P("No traces with errors found", style="color: #64748b; text-align: center; padding: 40px;")
            ),
            cls="main-container"
        )
    )


@rt('/traces')
def get(page: int = 1, limit: int = 50):
    logger.info(f"Loading all traces page {page}")

    from analyzer import get_db_connection

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get total count
    cursor.execute('SELECT COUNT(*) FROM traces')
    total = cursor.fetchone()[0]

    # Get paginated traces
    offset = (page - 1) * limit
    cursor.execute('''
        SELECT trace_id, model, num_turns, has_tool_calls, cost, total_tokens,
               potential_error_indicators, user_disagreement_detected
        FROM traces
        ORDER BY parsed_timestamp DESC
        LIMIT ? OFFSET ?
    ''', (limit, offset))

    traces = []
    for row in cursor.fetchall():
        traces.append({
            'trace_id': row[0],
            'model': row[1],
            'num_turns': row[2],
            'has_tool_calls': bool(row[3]),
            'cost': row[4],
            'total_tokens': row[5],
            'potential_error_indicators': row[6],
            'user_disagreement_detected': bool(row[7])
        })

    conn.close()

    total_pages = (total + limit - 1) // limit

    return Div(
        NavBar("traces"),
        Div(
            A("← Back to Dashboard", href="/", cls="back-link"),
            Div(
                H2("📋 All Traces", style="margin: 0;"),
                P(f"Showing {offset + 1}-{min(offset + limit, total)} of {total:,} traces", style="margin: 4px 0 0 0; color: #94a3b8;"),
                cls="section-header"
            ),
            *[TraceListItem(trace) for trace in traces],
            # Pagination
            Div(
                A("← Previous",
                  href=f"/traces?page={page-1}" if page > 1 else None,
                  cls=f"page-btn {'page-btn-active' if page > 1 else 'page-btn-disabled'}",
                  style="margin-right: 12px;"),
                Span(f"Page {page} of {total_pages}", style="color: #94a3b8; font-weight: 500;"),
                A("Next →",
                  href=f"/traces?page={page+1}" if page < total_pages else None,
                  cls=f"page-btn {'page-btn-active' if page < total_pages else 'page-btn-disabled'}",
                  style="margin-left: 12px;"),
                style="display: flex; align-items: center; justify-content: center; margin-top: 24px; padding: 16px;"
            ),
            cls="main-container"
        )
    )


if __name__ == "__main__":
    logger.info(f"Starting TraceAudit server on {HOST}:{PORT}")
    serve(host=HOST, port=PORT)
