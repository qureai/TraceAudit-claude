from fasthtml.common import *
import json
import re


def format_content(content: str) -> NotStr:
    """Format content with markdown-style formatting."""
    if not content:
        return NotStr("")
    content = content.replace('\\n', '\n')
    content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    return NotStr('<br>'.join(content.split('\n')))


def StatCard(title: str, value, subtitle: str = None, color: str = "#3b82f6"):
    return Div(
        Div(title, style="font-size: 0.85em; color: #94a3b8; margin-bottom: 4px;"),
        Div(str(value), style=f"font-size: 1.8em; font-weight: 700; color: {color};"),
        Div(subtitle, style="font-size: 0.75em; color: #64748b; margin-top: 4px;") if subtitle else None,
        style="background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748;"
    )


def ModelDistributionChart(model_distribution: dict):
    total = sum(model_distribution.values())
    bars = []
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899']
    for i, (model, count) in enumerate(model_distribution.items()):
        pct = (count / total * 100) if total > 0 else 0
        color = colors[i % len(colors)]
        bars.append(
            Div(
                Div(
                    Div(style=f"width: {pct}%; background: {color}; height: 100%; border-radius: 4px;"),
                    style="flex: 1; background: #0f0f1a; border-radius: 4px; height: 24px; overflow: hidden;"
                ),
                Div(f"{model.split('/')[-1]}", style="min-width: 150px; font-size: 0.85em; color: #e2e8f0;"),
                Div(f"{count:,} ({pct:.1f}%)", style="min-width: 120px; text-align: right; font-size: 0.85em; color: #94a3b8;"),
                style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;"
            )
        )
    return Div(
        H3("Model Distribution", style="margin-bottom: 16px; color: #e2e8f0;"),
        *bars,
        style="background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748;"
    )


def UseCaseCard(prompt_hash: str, info: dict, index: int):
    return A(
        Div(
            Div(
                Div(f"Use Case #{index + 1}", style="font-weight: 600; color: #e2e8f0; margin-bottom: 8px;"),
                Div(f"{info['count']:,} traces", style="font-size: 1.2em; font-weight: 700; color: #3b82f6;"),
                style="flex: 1;"
            ),
            Div(
                Div(info['description'][:80] + "..." if len(info.get('description', '')) > 80 else info.get('description', ''),
                    style="font-size: 0.85em; color: #94a3b8; margin-top: 12px; line-height: 1.4;"),
                Div(f"Primary model: {info.get('sample_model', 'N/A').split('/')[-1]}",
                    style="font-size: 0.75em; color: #64748b; margin-top: 8px;"),
            ),
            style="padding: 20px;"
        ),
        href=f"/use-case/{prompt_hash}",
        style="display: block; background: #16213e; border-radius: 12px; border: 1px solid #2d3748; text-decoration: none; transition: all 0.2s;",
        cls="use-case-card"
    )


def TraceListItem(trace: dict, show_use_case: bool = False):
    error_badge = None
    if trace.get('potential_error_indicators') or trace.get('user_disagreement_detected'):
        indicators = []
        if trace.get('user_disagreement_detected'):
            indicators.append("User Disagreement")
        if trace.get('potential_error_indicators'):
            indicators.extend(trace['potential_error_indicators'].split(','))
        error_badge = Div(
            *[Span(ind, style="background: #dc2626; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7em; margin-right: 4px;") for ind in indicators[:2]],
            style="margin-top: 8px;"
        )

    # Multi-element badge
    element_count = trace.get('element_count', 1)
    multi_element_badge = None
    if element_count > 1:
        multi_element_badge = Span(
            f"📦 {element_count} elements",
            style="background: #8b5cf6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7em; margin-left: 8px;"
        )

    return A(
        Div(
            Div(
                Div(
                    Span(f"Trace: {trace['trace_id'][:16]}...", style="font-weight: 600; color: #e2e8f0;"),
                    multi_element_badge,
                    style="display: flex; align-items: center;"
                ),
                Div(
                    Span(trace['model'].split('/')[-1], style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75em;"),
                    Span(f"{trace['num_turns']} turn{'s' if trace['num_turns'] > 1 else ''}", style="margin-left: 8px; color: #94a3b8; font-size: 0.85em;"),
                    Span("🔧" if trace.get('has_tool_calls') else "", style="margin-left: 8px;"),
                    style="margin-top: 4px;"
                ),
                style="flex: 1;"
            ),
            Div(
                Div(f"${trace.get('cost', 0):.4f}", style="font-size: 0.85em; color: #10b981;"),
                Div(f"{trace.get('total_tokens', 0):,} tokens", style="font-size: 0.75em; color: #64748b;"),
                style="text-align: right;"
            ),
            error_badge,
            style="display: flex; flex-wrap: wrap; align-items: flex-start; padding: 16px;"
        ),
        href=f"/trace/{trace['trace_id']}",
        style="display: block; background: #1f2940; border-radius: 8px; border: 1px solid #2d3748; text-decoration: none; margin-bottom: 8px; transition: all 0.2s;",
        cls="trace-item"
    )


def MessageBubble(msg: dict, index: int, model: str = None, turn_number: int = None):
    """Render a message bubble with role-based styling."""
    role = msg.get('role', 'unknown')
    content = msg.get('content', '')
    tool_calls = msg.get('tool_calls', [])

    # Role-based styling
    style_map = {
        'system': ('SYSTEM', '#2d2d3d', '4px solid #6b7280', '#94a3b8'),
        'user': ('USER', '#1e3a5f', '4px solid #3b82f6', '#e2e8f0'),
        'assistant': ('ASSISTANT', '#1f2940', '4px solid #10b981', '#e2e8f0'),
        'tool': ('TOOL', '#3d2e1f', '4px solid #f59e0b', '#e2e8f0'),
    }

    label, bg_color, border, text_color = style_map.get(role, ('UNKNOWN', '#1f2940', 'none', '#e2e8f0'))

    # Add turn number and model to label
    if turn_number and role in ['user', 'assistant']:
        label = f"{label} - Turn {turn_number}"
    if model and role == 'assistant':
        label = f"{label} ({model.split('/')[-1]})"

    # Handle system prompt placeholder
    if role == 'system':
        return Div(
            Div(
                Div(label, style="display: inline-block; font-size: 0.75em; font-weight: 600; color: #94a3b8; text-transform: uppercase; margin-right: 10px;"),
                Span("▶", id=f"sys-toggle-{index}", style="cursor: pointer; user-select: none; color: #6b7280; font-size: 0.9em;"),
                style="cursor: pointer;",
                onclick=f"toggleSystemMsg('sys-content-{index}', 'sys-toggle-{index}')"
            ),
            Div(
                Pre(content[:500] + "..." if len(content) > 500 else content,
                    style="white-space: pre-wrap; font-size: 0.85em; margin: 0; color: #94a3b8;"),
                Div("[System prompt replaced with placeholder - see right panel for full content]",
                    style="margin-top: 8px; font-style: italic; color: #64748b; font-size: 0.8em;"),
                id=f"sys-content-{index}",
                style="display: none; margin-top: 8px;"
            ),
            style=f"background: {bg_color}; border-left: {border}; padding: 16px; border-radius: 8px; margin: 12px 0;"
        )

    # Handle tool calls
    if tool_calls:
        tool_elements = []
        for tc in tool_calls:
            func = tc.get('function', {})
            tool_name = func.get('name', 'unknown')
            try:
                args = json.loads(func.get('arguments', '{}'))
                args_str = json.dumps(args, indent=2)
            except:
                args_str = func.get('arguments', '')

            tool_elements.append(
                Div(
                    Div(
                        Span("🔧", style="margin-right: 8px;"),
                        Span(tool_name, style="font-weight: 600; color: #f59e0b;"),
                        style="margin-bottom: 8px;"
                    ),
                    Pre(args_str[:300] + "..." if len(args_str) > 300 else args_str,
                        style="white-space: pre-wrap; font-size: 0.8em; background: #0f0f1a; padding: 8px; border-radius: 4px; margin: 0; color: #94a3b8;"),
                    style="margin-top: 8px; padding: 12px; background: #2d2d1f; border-radius: 6px;"
                )
            )

        return Div(
            Div(label, style="font-size: 0.75em; font-weight: 600; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;"),
            Div(format_content(content) if content else None, style=f"color: {text_color}; line-height: 1.6;") if content else None,
            *tool_elements,
            style=f"background: {bg_color}; border-left: {border}; padding: 16px; border-radius: 8px; margin: 12px 0;"
        )

    # Handle tool response
    if role == 'tool':
        return Div(
            Div("TOOL RESPONSE", style="font-size: 0.75em; font-weight: 600; color: #f59e0b; text-transform: uppercase; margin-bottom: 8px;"),
            Pre(content[:500] + "..." if len(content) > 500 else content,
                style="white-space: pre-wrap; font-size: 0.8em; margin: 0; color: #94a3b8; background: #0f0f1a; padding: 8px; border-radius: 4px;"),
            style=f"background: {bg_color}; border-left: {border}; padding: 16px; border-radius: 8px; margin: 12px 0;"
        )

    # Regular message
    return Div(
        Div(label, style="font-size: 0.75em; font-weight: 600; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;"),
        Div(format_content(content), style=f"color: {text_color}; line-height: 1.6;"),
        style=f"background: {bg_color}; border-left: {border}; padding: 16px; border-radius: 8px; margin: 12px 0;"
    )


def ErrorDetectionPanel(stats: dict):
    return Div(
        H3("⚠️ Error Detection Summary", style="margin-bottom: 16px; color: #f87171;"),
        Div(
            StatCard("User Disagreements", stats.get('user_disagreements', 0), "Multi-turn conversations", "#ef4444"),
            StatCard("Error Indicators", stats.get('traces_with_error_indicators', 0), "Potential issues detected", "#f59e0b"),
            style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 16px;"
        ),
        Div(
            H4("Error Indicator Breakdown", style="margin-bottom: 12px; color: #e2e8f0; font-size: 0.9em;"),
            *[
                Div(
                    Span(indicator, style="flex: 1; color: #e2e8f0;"),
                    Span(str(count), style="color: #f87171; font-weight: 600;"),
                    style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #2d3748;"
                )
                for indicator, count in stats.get('error_indicator_breakdown', {}).items()
            ] if stats.get('error_indicator_breakdown') else [Div("No error indicators found", style="color: #64748b;")],
            style="background: #0f0f1a; padding: 16px; border-radius: 8px;"
        ),
        A("View All Traces with Errors →", href="/errors", style="display: block; margin-top: 16px; color: #3b82f6; text-decoration: none;"),
        style="background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748;"
    )


def MetadataStatsPanel(stats: dict):
    return Div(
        H3("📋 Metadata Analysis", style="margin-bottom: 16px; color: #e2e8f0;"),
        Div(
            StatCard("With Metadata", stats.get('with_metadata', 0), color="#10b981"),
            StatCard("Without Metadata", stats.get('without_metadata', 0), color="#6b7280"),
            style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 16px;"
        ),
        Div(
            H4("Metadata Keys Found", style="margin-bottom: 12px; color: #e2e8f0; font-size: 0.9em;"),
            *[
                Div(
                    Span(key, style="flex: 1; color: #94a3b8; font-size: 0.85em;"),
                    Span(str(count), style="color: #3b82f6; font-weight: 600;"),
                    style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #2d3748;"
                )
                for key, count in stats.get('unique_metadata_keys', {}).items()
            ] if stats.get('unique_metadata_keys') else [Div("No metadata keys found", style="color: #64748b;")],
            style="background: #0f0f1a; padding: 16px; border-radius: 8px; max-height: 300px; overflow-y: auto;"
        ),
        style="background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748;"
    )


def TokenCostPanel(token_stats: dict, cost_stats: dict, response_time_stats: dict):
    return Div(
        H3("📊 Performance Metrics", style="margin-bottom: 16px; color: #e2e8f0;"),
        Div(
            StatCard("Total Tokens", f"{token_stats.get('total_tokens', 0):,}", color="#8b5cf6"),
            StatCard("Avg Tokens/Request", f"{token_stats.get('avg_tokens', 0):,.0f}", color="#8b5cf6"),
            StatCard("Total Cost", f"${cost_stats.get('total_cost', 0):.2f}", color="#10b981"),
            StatCard("Avg Response Time", f"{response_time_stats.get('avg_response_time_ms', 0):,.0f}ms", color="#f59e0b"),
            style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;"
        ),
        style="background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748;"
    )


def NavBar(active: str = "dashboard"):
    links = [
        ("Dashboard", "/", "dashboard"),
        ("Use Cases", "/use-cases", "use-cases"),
        ("Multi-Element", "/multi-element", "multi-element"),
        ("Errors", "/errors", "errors"),
        ("All Traces", "/traces", "traces"),
    ]
    return Div(
        Div(
            H1("🔍 TraceAudit", style="margin: 0; font-size: 1.5em; color: #e2e8f0;"),
            style="display: flex; align-items: center;"
        ),
        Div(
            *[
                A(
                    label,
                    href=href,
                    style=f"padding: 8px 16px; border-radius: 6px; text-decoration: none; color: {'#3b82f6' if key == active else '#94a3b8'}; background: {'#1f2940' if key == active else 'transparent'}; font-weight: {'600' if key == active else '400'};"
                )
                for label, href, key in links
            ],
            style="display: flex; gap: 8px;"
        ),
        style="display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; background: #16213e; border-bottom: 1px solid #2d3748;"
    )


def MultiElementStatsPanel(stats: dict):
    return Div(
        H3("📦 Multi-Element Traces", style="margin-bottom: 16px; color: #8b5cf6;"),
        Div(
            StatCard("Traces with Multiple Elements", stats.get('traces_with_multiple_elements', 0), color="#8b5cf6"),
            StatCard("Max Elements/Trace", stats.get('max_elements_per_trace', 1), color="#ec4899"),
            StatCard("Avg Elements/Trace", stats.get('avg_elements_per_trace', 1), color="#f59e0b"),
            StatCard("Extra Elements", stats.get('total_extra_elements', 0), "Beyond unique trace IDs", "#6b7280"),
            style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;"
        ),
        A("View All Multi-Element Traces →", href="/multi-element", style="display: block; margin-top: 16px; color: #8b5cf6; text-decoration: none;"),
        style="background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748;"
    )


def FilterPanel(filters: dict, current_model: str = None, current_use_case: str = None, has_errors: bool = None, multi_element_only: bool = False):
    return Div(
        H4("Filters", style="margin-bottom: 12px; color: #e2e8f0;"),
        Form(
            Div(
                Label("Model:", style="color: #94a3b8; font-size: 0.85em;"),
                Select(
                    Option("All Models", value="", selected=not current_model),
                    *[Option(m.split('/')[-1], value=m, selected=m == current_model) for m in filters.get('models', [])],
                    name="model",
                    style="width: 100%; padding: 8px; background: #0f0f1a; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;"
                ),
                style="margin-bottom: 12px;"
            ),
            Div(
                Label("Use Case:", style="color: #94a3b8; font-size: 0.85em;"),
                Select(
                    Option("All Use Cases", value="", selected=not current_use_case),
                    *[Option(f"{uc['hash'][:8]}... ({uc['count']})", value=uc['hash'], selected=uc['hash'] == current_use_case) for uc in filters.get('use_cases', [])],
                    name="use_case",
                    style="width: 100%; padding: 8px; background: #0f0f1a; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;"
                ),
                style="margin-bottom: 12px;"
            ),
            Div(
                Label(
                    Input(type="checkbox", name="has_errors", value="true", checked=has_errors, style="margin-right: 8px;"),
                    Span("With Errors Only", style="color: #94a3b8; font-size: 0.85em;"),
                ),
                style="margin-bottom: 12px;"
            ),
            Div(
                Label(
                    Input(type="checkbox", name="multi_element_only", value="true", checked=multi_element_only, style="margin-right: 8px;"),
                    Span("Multi-Element Only", style="color: #94a3b8; font-size: 0.85em;"),
                ),
                style="margin-bottom: 12px;"
            ),
            Button("Apply Filters", type="submit", style="width: 100%; padding: 10px; background: #3b82f6; color: white; border: none; border-radius: 6px; cursor: pointer;"),
            A("Clear Filters", href="/traces", style="display: block; text-align: center; margin-top: 8px; color: #64748b; font-size: 0.85em;"),
            method="get",
            action="/traces"
        ),
        style="background: #1f2940; padding: 16px; border-radius: 12px; border: 1px solid #2d3748; margin-bottom: 16px;"
    )


def ElementCard(element: dict, index: int, total: int):
    """Card for displaying a single trace element in a multi-element trace view."""
    return Div(
        Div(
            Div(
                Span(f"Element {index + 1} of {total}", style="font-weight: 600; color: #e2e8f0;"),
                Span(element.get('model', 'unknown').split('/')[-1], style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; margin-left: 12px;"),
                style="display: flex; align-items: center;"
            ),
            Div(element.get('created_at', 'N/A'), style="font-size: 0.8em; color: #64748b; margin-top: 4px;"),
            style="flex: 1;"
        ),
        Div(
            Div(f"${element.get('cost', 0):.4f}", style="font-size: 0.85em; color: #10b981;"),
            Div(f"{element.get('total_tokens', 0):,} tokens", style="font-size: 0.75em; color: #64748b;"),
            Div(f"{element.get('num_turns', 0)} turns", style="font-size: 0.75em; color: #94a3b8;"),
            style="text-align: right;"
        ),
        style="display: flex; justify-content: space-between; padding: 16px; background: #16213e; border-radius: 8px; border: 1px solid #2d3748; margin-bottom: 8px; cursor: pointer;",
        onclick=f"showElement({index})"
    )
