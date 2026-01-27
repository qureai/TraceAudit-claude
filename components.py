from fasthtml.common import *
import json
import re
from datetime import datetime


def format_date(date_str: str) -> str:
    """Format date to YYYY-MM-DD HH:MM:SS format."""
    if not date_str or date_str == 'Unknown' or date_str == 'N/A':
        return date_str or 'N/A'
    try:
        # Try parsing ISO format
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(date_str)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        # If parsing fails, try to extract just the date/time portion
        if len(date_str) >= 19:
            return date_str[:19].replace('T', ' ')
        return date_str


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


def PieChart(data: dict, title: str, chart_id: str):
    """Create a pie chart using CSS conic-gradient."""
    total = sum(data.values())
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']

    # Build conic gradient
    gradient_parts = []
    cumulative = 0
    legend_items = []

    for i, (label, count) in enumerate(data.items()):
        pct = (count / total * 100) if total > 0 else 0
        color = colors[i % len(colors)]
        start_pct = cumulative
        cumulative += pct
        gradient_parts.append(f"{color} {start_pct}% {cumulative}%")

        # Short label for display
        short_label = label.split('/')[-1] if '/' in label else label
        if len(short_label) > 20:
            short_label = short_label[:17] + "..."

        legend_items.append(
            Div(
                Div(style=f"width: 12px; height: 12px; background: {color}; border-radius: 2px; flex-shrink: 0;"),
                Div(
                    Div(short_label, style="font-size: 0.8em; color: #e2e8f0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"),
                    Div(f"{count:,} ({pct:.1f}%)", style="font-size: 0.7em; color: #64748b;"),
                    style="flex: 1; min-width: 0;"
                ),
                style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;"
            )
        )

    gradient = ", ".join(gradient_parts) if gradient_parts else "#2d3748 0% 100%"

    return Div(
        H4(title, style="margin-bottom: 12px; color: #e2e8f0; font-size: 0.95em;"),
        Div(
            # Pie chart
            Div(
                style=f"width: 140px; height: 140px; border-radius: 50%; background: conic-gradient({gradient}); flex-shrink: 0;"
            ),
            # Legend
            Div(
                *legend_items,
                style="flex: 1; max-height: 160px; overflow-y: auto;"
            ),
            style="display: flex; gap: 16px; align-items: center;"
        ),
        style="background: #16213e; padding: 16px; border-radius: 8px; flex: 1;"
    )


def ModelDistributionChart(model_distribution: dict):
    return PieChart(model_distribution, "Model Distribution", "model-pie")


def UseCaseDistributionChart(use_case_distribution: dict):
    """Create a pie chart for use case distribution."""
    # Convert to simple count dict using actual use case names
    simple_data = {}
    for hash_val, info in list(use_case_distribution.items())[:8]:  # Top 8 use cases
        # Use actual use case name if available, otherwise fallback to hash prefix
        use_case_name = info.get('use_case_name')
        if use_case_name:
            # Truncate long names for the chart
            label = use_case_name[:20] + "..." if len(use_case_name) > 20 else use_case_name
        else:
            label = f"{hash_val[:8]}..."
        simple_data[label] = info['count']
    return PieChart(simple_data, "Use Case Distribution", "usecase-pie")


def UseCaseCard(prompt_hash: str, info: dict, index: int, workspace_stats: list = None):
    """
    Render a use case card.

    Args:
        prompt_hash: The system prompt hash
        info: Use case info dict from analysis
        index: Index for numbering
        workspace_stats: List of {'workspace': str, 'region': str, 'count': int} from metadata
    """
    # Use actual use case name if available
    use_case_name = info.get('use_case_name')
    title = use_case_name if use_case_name else f"Use Case #{index + 1}"

    # Type badges - create separate badge for each type
    use_case_type = info.get('use_case_type', '')
    type_badges = []
    type_colors = {'workflows': '#8b5cf6', 'agents': '#10b981', 'prompts': '#f59e0b'}
    if use_case_type:
        for t in use_case_type.split(', '):
            t = t.strip()
            if t:
                color = type_colors.get(t, '#6b7280')
                type_badges.append(
                    Span(t, style=f"background: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7em;")
                )

    # Version badge
    version = info.get('use_case_version', '')
    version_badge = None
    if version:
        version_badge = Span(version, style="background: #374151; color: #d1d5db; padding: 2px 8px; border-radius: 4px; font-size: 0.7em;")

    # Build workspace display from metadata stats
    workspace_display = None
    if workspace_stats:
        # Format: (region)workspace(count), ...
        # Region colors
        region_colors = {'US': '#3b82f6', 'EU': '#8b5cf6', 'IN': '#f59e0b', '?': '#6b7280'}

        workspace_parts = []
        for ws in workspace_stats[:5]:  # Limit to top 5 workspaces
            region = ws.get('region', '?')
            workspace_name = ws.get('workspace', 'unknown')
            count = ws.get('count', 0)
            color = region_colors.get(region, '#6b7280')

            # Add title attribute for hover tooltip with full name
            is_truncated = len(workspace_name) > 15
            display_name = f"{workspace_name[:15]}..." if is_truncated else workspace_name

            workspace_parts.append(
                Span(
                    Span(f"({region})", style=f"color: {color}; font-weight: 600;"),
                    Span(f"{display_name}({count})", style="color: #94a3b8;"),
                    title=f"{workspace_name} ({region}) - {count} traces",  # Hover tooltip
                    style="margin-right: 8px; font-size: 0.75em; cursor: help;"
                )
            )

        if len(workspace_stats) > 5:
            # Build tooltip text for remaining workspaces
            remaining = workspace_stats[5:]
            remaining_text = "\n".join([
                f"({ws.get('region', '?')}) {ws.get('workspace', 'unknown')}: {ws.get('count', 0)} traces"
                for ws in remaining
            ])
            workspace_parts.append(
                Span(
                    f"+{len(remaining)} more",
                    title=remaining_text,  # Hover tooltip with all remaining
                    style="color: #64748b; font-size: 0.7em; cursor: help; text-decoration: underline dotted;"
                )
            )

        workspace_display = Div(
            *workspace_parts,
            style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px; align-items: center;"
        )
    else:
        # Fallback to old workspace info from prompts table
        workspace = info.get('workspace', '')
        if workspace:
            workspace_display = Div(f"Workspace: {workspace}", style="font-size: 0.75em; color: #64748b; margin-top: 8px;")

    return A(
        Div(
            Div(
                Div(
                    Span(title, style="font-weight: 600; color: #e2e8f0;"),
                    *type_badges,
                    version_badge,
                    style="display: flex; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 4px;"
                ),
                Div(f"{info['count']:,} traces", style="font-size: 1.2em; font-weight: 700; color: #3b82f6;"),
                style="flex: 1;"
            ),
            Div(
                Div(info['description'][:80] + "..." if len(info.get('description', '')) > 80 else info.get('description', ''),
                    style="font-size: 0.85em; color: #94a3b8; margin-top: 12px; line-height: 1.4;") if not use_case_name else None,
                workspace_display,
                Div(f"Model: {info.get('sample_model', 'N/A').split('/')[-1]}",
                    style="font-size: 0.75em; color: #64748b; margin-top: 4px;"),
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

    # Multi-trace badge
    element_count = trace.get('element_count', 1)
    multi_element_badge = None
    if element_count > 1:
        multi_element_badge = Span(
            f"📦 {element_count} traces",
            style="background: #8b5cf6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7em; margin-left: 8px;"
        )

    # Patient/workspace info badges
    patient_info = []
    if trace.get('workspace_name'):
        patient_info.append(
            Span(trace['workspace_name'], style="background: #059669; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7em; margin-right: 4px;")
        )
    if trace.get('patient_name'):
        patient_info.append(
            Span(f"🏥 {trace['patient_name'][:20]}{'...' if len(trace.get('patient_name', '')) > 20 else ''}", style="color: #94a3b8; font-size: 0.75em;")
        )

    patient_row = None
    if patient_info:
        patient_row = Div(*patient_info, style="margin-top: 4px; display: flex; align-items: center; gap: 4px;")

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
                patient_row,
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
        ("Multi-Trace", "/multi-element", "multi-element"),
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
        H3("📦 Multi-Trace Groups", style="margin-bottom: 16px; color: #8b5cf6;"),
        Div(
            StatCard("Groups with Multiple Traces", stats.get('traces_with_multiple_elements', 0), color="#8b5cf6"),
            StatCard("Max Traces/Group", stats.get('max_elements_per_trace', 1), color="#ec4899"),
            StatCard("Avg Traces/Group", stats.get('avg_elements_per_trace', 1), color="#f59e0b"),
            StatCard("Extra Traces", stats.get('total_extra_elements', 0), "Beyond unique trace IDs", "#6b7280"),
            style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;"
        ),
        A("View All Multi-Trace Groups →", href="/multi-element", style="display: block; margin-top: 16px; color: #8b5cf6; text-decoration: none;"),
        style="background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748;"
    )


def FilterPanel(filters: dict, current_model: str = None, current_use_case: str = None, has_errors: bool = None, multi_element_only: bool = False, with_metadata: bool = None, multi_turn: bool = None, current_workspace: str = None, current_error_status: str = None, current_error_check: str = None, available_error_checks: list = None):
    # Get workspaces from filters
    workspaces = filters.get('workspaces', [])

    # Build error check options
    error_check_options = []
    if available_error_checks:
        for check in available_error_checks:
            check_id = check.get('check_id', '')
            errors = check.get('errors', 0)
            error_check_options.append(
                Option(f"{check_id} ({errors})", value=check_id, selected=check_id == current_error_check)
            )

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
                    *[Option(
                        f"{uc['name']} ({uc['count']})" if uc.get('name') else f"{uc['hash'][:8]}... ({uc['count']})",
                        value=uc['hash'],
                        selected=uc['hash'] == current_use_case
                    ) for uc in filters.get('use_cases', [])],
                    name="use_case",
                    style="width: 100%; padding: 8px; background: #0f0f1a; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;"
                ),
                style="margin-bottom: 12px;"
            ),
            # Workspace filter
            Div(
                Label("Workspace:", style="color: #94a3b8; font-size: 0.85em;"),
                Select(
                    Option("All Workspaces", value="", selected=not current_workspace),
                    *[Option(f"{w['workspace_name']} ({w['count']})", value=w['workspace_name'], selected=w['workspace_name'] == current_workspace) for w in workspaces],
                    name="workspace",
                    style="width: 100%; padding: 8px; background: #0f0f1a; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;"
                ),
                style="margin-bottom: 12px;"
            ) if workspaces else None,
            Div(
                Label(
                    Input(type="checkbox", name="has_errors", value="true", checked=has_errors, style="margin-right: 8px;"),
                    Span("With Errors Only", style="color: #94a3b8; font-size: 0.85em;"),
                ),
                style="margin-bottom: 12px;"
            ),
            Div(
                Label(
                    Input(type="checkbox", name="with_metadata", value="true", checked=with_metadata, style="margin-right: 8px;"),
                    Span("With Metadata", style="color: #94a3b8; font-size: 0.85em;"),
                ),
                style="margin-bottom: 12px;"
            ),
            Div(
                Label(
                    Input(type="checkbox", name="multi_turn", value="true", checked=multi_turn, style="margin-right: 8px;"),
                    Span("Multi-turn", style="color: #94a3b8; font-size: 0.85em;"),
                ),
                style="margin-bottom: 12px;"
            ),
            Div(
                Label(
                    Input(type="checkbox", name="multi_element_only", value="true", checked=multi_element_only, style="margin-right: 8px;"),
                    Span("Multi-Trace Only", style="color: #94a3b8; font-size: 0.85em;"),
                ),
                style="margin-bottom: 12px;"
            ),
            # Error Detection Filters section
            Div(
                H4("Error Detection", style="margin-bottom: 12px; color: #f59e0b; font-size: 0.9em;"),
                Div(
                    Label("Error Status:", style="color: #94a3b8; font-size: 0.85em;"),
                    Select(
                        Option("All", value="", selected=not current_error_status),
                        Option("Has Errors", value="has_errors", selected=current_error_status == "has_errors"),
                        Option("No Errors", value="no_errors", selected=current_error_status == "no_errors"),
                        Option("Not Checked", value="not_checked", selected=current_error_status == "not_checked"),
                        name="error_status",
                        style="width: 100%; padding: 8px; background: #0f0f1a; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;"
                    ),
                    style="margin-bottom: 12px;"
                ),
                Div(
                    Label("Error Check:", style="color: #94a3b8; font-size: 0.85em;"),
                    Select(
                        Option("All Checks", value="", selected=not current_error_check),
                        *error_check_options,
                        name="error_check",
                        style="width: 100%; padding: 8px; background: #0f0f1a; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;"
                    ),
                    style="margin-bottom: 12px;"
                ) if error_check_options else None,
                style="background: #2d2d1f; padding: 12px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #f59e0b;"
            ) if available_error_checks else None,
            Button("Apply Filters", type="submit", style="width: 100%; padding: 10px; background: #3b82f6; color: white; border: none; border-radius: 6px; cursor: pointer;"),
            A("Clear Filters", href="/traces", style="display: block; text-align: center; margin-top: 8px; color: #64748b; font-size: 0.85em;"),
            method="get",
            action="/traces"
        ),
        style="background: #1f2940; padding: 16px; border-radius: 12px; border: 1px solid #2d3748; margin-bottom: 16px;"
    )


def ElementCard(element: dict, index: int, total: int):
    """Card for displaying a single trace in a multi-trace view."""
    return Div(
        Div(
            Div(
                Span(f"Trace {index + 1} of {total}", style="font-weight: 600; color: #e2e8f0;"),
                Span(element.get('model', 'unknown').split('/')[-1], style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; margin-left: 12px;"),
                style="display: flex; align-items: center;"
            ),
            Div(format_date(element.get('created_at', 'N/A')), style="font-size: 0.8em; color: #64748b; margin-top: 4px;"),
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


def PatientMetadataPanel(metadata: dict):
    """Panel to display patient/workspace metadata in trace detail view."""
    if not metadata:
        return None

    match_status = "Matched" if metadata.get('match_success') else "Not Matched"
    match_color = "#10b981" if metadata.get('match_success') else "#ef4444"

    return Div(
        H4("Patient Information", style="margin-bottom: 12px; color: #059669; font-size: 0.9em;"),
        Div(
            # Match status badge
            Span(match_status, style=f"background: {match_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; margin-bottom: 8px; display: inline-block;"),
            # Patient details
            Div(
                Div(f"Patient ID: {metadata.get('patient_id') or 'N/A'}", style="font-size: 0.85em; color: #e2e8f0; margin-bottom: 4px;"),
                Div(f"Patient Name: {metadata.get('patient_name') or 'N/A'}", style="font-size: 0.85em; color: #e2e8f0; margin-bottom: 4px;"),
                Div(f"Patient PK: {metadata.get('patient_pk') or 'N/A'}", style="font-size: 0.85em; color: #64748b; margin-bottom: 4px;") if metadata.get('patient_pk') else None,
                style="margin-top: 8px;"
            ),
            # Workspace details
            Div(
                Div(f"Workspace: {metadata.get('workspace_name') or 'N/A'}", style="font-size: 0.85em; color: #10b981; font-weight: 600; margin-bottom: 4px;"),
                Div(f"Workspace ID: {metadata.get('workspace_id') or 'N/A'}", style="font-size: 0.85em; color: #64748b; margin-bottom: 4px;") if metadata.get('workspace_id') else None,
                Div(f"Replica: {metadata.get('replica_source') or 'N/A'}", style="font-size: 0.85em; color: #64748b; margin-bottom: 4px;") if metadata.get('replica_source') else None,
                style="margin-top: 8px;"
            ),
            # Match error if any
            Div(f"Note: {metadata.get('match_error')}", style="font-size: 0.8em; color: #f59e0b; margin-top: 8px;") if metadata.get('match_error') else None,
        ),
        style="background: #0f2f1a; padding: 12px; border-radius: 8px; margin-bottom: 16px; border: 1px solid #059669;"
    )


def ErrorCheckResultsPanel(error_summary: dict, element_id: str = None):
    """
    Panel to display error check results for a trace in the detail view.

    Args:
        error_summary: dict with total_checks, errors_found, checked, error_details
        element_id: Optional element_id for rerun button
    """
    if not error_summary.get('checked'):
        return Div(
            H4("Error Detection", style="margin-bottom: 12px; color: #f59e0b; font-size: 0.9em;"),
            Div(
                Div("Not Checked", style="color: #64748b; margin-bottom: 8px;"),
                Div("Run error detection to check this trace", style="font-size: 0.85em; color: #94a3b8;"),
                Form(
                    Button(
                        "Run Checks",
                        type="submit",
                        style="padding: 8px 16px; background: #f59e0b; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85em; margin-top: 8px;"
                    ),
                    action=f"/run-error-check/{element_id}" if element_id else "#",
                    method="post"
                ) if element_id else None,
                style="text-align: center; padding: 16px;"
            ),
            style="background: #2d2d1f; padding: 12px; border-radius: 8px; margin-bottom: 16px; border: 1px solid #f59e0b;"
        )

    errors_found = error_summary.get('errors_found', 0)
    total_checks = error_summary.get('total_checks', 0)
    error_details = error_summary.get('error_details', [])

    # Determine status color
    if errors_found == 0:
        status_color = '#10b981'
        status_text = 'All Checks Passed'
        border_color = '#10b981'
        bg_color = '#0f2f1a'
    else:
        status_color = '#ef4444'
        status_text = f'{errors_found} Error{"s" if errors_found > 1 else ""} Found'
        border_color = '#ef4444'
        bg_color = '#2f1f1f'

    # Build check results list
    check_items = []
    for check in error_details:
        check_id = check.get('check_id', '')
        check_level = check.get('check_level', 1)
        has_error = check.get('has_error', False)
        reason = check.get('error_reason', '')

        # Level colors
        level_color = '#ef4444' if check_level == 1 else '#8b5cf6'
        level_label = f"L{check_level}"

        # Status indicator
        if has_error:
            status_icon = "❌"
            item_color = '#f87171'
        else:
            status_icon = "✓"
            item_color = '#4ade80'

        check_items.append(
            Div(
                Div(
                    Span(status_icon, style=f"margin-right: 8px; color: {item_color};"),
                    Span(level_label, style=f"background: {level_color}; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.7em; margin-right: 8px;"),
                    Span(check_id, style=f"color: {'#f87171' if has_error else '#94a3b8'}; font-size: 0.85em;"),
                    style="display: flex; align-items: center;"
                ),
                Div(reason, style="font-size: 0.8em; color: #f59e0b; margin-top: 4px; margin-left: 28px;") if has_error and reason else None,
                style=f"padding: 8px 0; border-bottom: 1px solid #2d3748;"
            )
        )

    return Div(
        H4("Error Detection Results", style="margin-bottom: 12px; color: #f59e0b; font-size: 0.9em;"),
        # Status badge
        Div(
            Span(status_text, style=f"background: {status_color}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 0.85em;"),
            Span(f"{total_checks} checks run", style="color: #64748b; font-size: 0.85em; margin-left: 12px;"),
            style="margin-bottom: 12px;"
        ),
        # Check results
        Div(
            *check_items,
            style="max-height: 200px; overflow-y: auto;"
        ) if check_items else None,
        # Rerun button
        Form(
            Button(
                "Re-run Checks",
                type="submit",
                style="padding: 6px 12px; background: #6b7280; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.8em; margin-top: 8px;"
            ),
            action=f"/run-error-check/{element_id}" if element_id else "#",
            method="post"
        ) if element_id else None,
        style=f"background: {bg_color}; padding: 12px; border-radius: 8px; margin-bottom: 16px; border: 1px solid {border_color};"
    )


def ErrorDetectionStatsPanel(stats: dict):
    """Panel showing error detection stats for the dashboard."""
    if not stats or stats.get('total_checked', 0) == 0:
        return Div(
            H3("🔍 Error Detection", style="margin-bottom: 16px; color: #f59e0b;"),
            Div(
                P("No traces have been checked yet.", style="color: #64748b; margin-bottom: 12px;"),
                Form(
                    Button(
                        "Run Error Detection",
                        type="submit",
                        style="padding: 10px 20px; background: #f59e0b; color: white; border: none; border-radius: 8px; cursor: pointer;"
                    ),
                    action="/run-error-checks",
                    method="post"
                ),
                style="text-align: center; padding: 20px;"
            ),
            style="background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748;"
        )

    total_checked = stats.get('total_checked', 0)
    traces_with_errors = stats.get('traces_with_errors', 0)
    error_rate = stats.get('error_rate', 0)
    by_check = stats.get('by_check', [])

    return Div(
        H3("🔍 Error Detection", style="margin-bottom: 16px; color: #f59e0b;"),
        Div(
            StatCard("Traces Checked", f"{total_checked:,}", color="#3b82f6"),
            StatCard("With Errors", traces_with_errors, f"{error_rate}% error rate", "#ef4444"),
            style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 16px;"
        ),
        Div(
            H4("Error Breakdown", style="margin-bottom: 12px; color: #e2e8f0; font-size: 0.9em;"),
            *[
                Div(
                    Div(
                        Span(f"L{check.get('check_level', 1)}", style=f"background: {'#ef4444' if check.get('check_level') == 1 else '#8b5cf6'}; color: white; padding: 1px 6px; border-radius: 3px; font-size: 0.7em; margin-right: 8px;"),
                        Span(check.get('check_id', ''), style="flex: 1; color: #e2e8f0; font-size: 0.85em;"),
                        style="display: flex; align-items: center; flex: 1;"
                    ),
                    Span(f"{check.get('errors', 0)}/{check.get('total', 0)}", style="color: #f87171; font-weight: 600; font-size: 0.85em;"),
                    style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #2d3748;"
                )
                for check in by_check[:8]
            ] if by_check else [Div("No checks recorded", style="color: #64748b;")],
            style="background: #0f0f1a; padding: 16px; border-radius: 8px;"
        ),
        Div(
            Form(
                Button(
                    "Run Error Detection",
                    type="submit",
                    style="padding: 8px 16px; background: #f59e0b; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85em; margin-right: 8px;"
                ),
                action="/run-error-checks",
                method="post",
                style="display: inline;"
            ),
            A("View Details →", href="/error-detection", style="color: #3b82f6; font-size: 0.85em;"),
            style="margin-top: 16px; display: flex; align-items: center; gap: 16px;"
        ),
        style="background: #1f2940; padding: 20px; border-radius: 12px; border: 1px solid #2d3748;"
    )


def ErrorDetectionFilterPanel(filters: dict, current_error_status: str = None, current_error_check: str = None, available_checks: list = None):
    """Filter panel for error detection filters on traces page."""
    check_options = []
    if available_checks:
        for check in available_checks:
            check_id = check.get('check_id', '')
            errors = check.get('errors', 0)
            check_options.append(
                Option(f"{check_id} ({errors})", value=check_id, selected=check_id == current_error_check)
            )

    return Div(
        H4("Error Detection Filters", style="margin-bottom: 12px; color: #f59e0b;"),
        Div(
            Label("Error Status:", style="color: #94a3b8; font-size: 0.85em;"),
            Select(
                Option("All", value="", selected=not current_error_status),
                Option("Has Errors", value="has_errors", selected=current_error_status == "has_errors"),
                Option("No Errors", value="no_errors", selected=current_error_status == "no_errors"),
                Option("Not Checked", value="not_checked", selected=current_error_status == "not_checked"),
                name="error_status",
                style="width: 100%; padding: 8px; background: #0f0f1a; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;"
            ),
            style="margin-bottom: 12px;"
        ),
        Div(
            Label("Error Check:", style="color: #94a3b8; font-size: 0.85em;"),
            Select(
                Option("All Checks", value="", selected=not current_error_check),
                *check_options,
                name="error_check",
                style="width: 100%; padding: 8px; background: #0f0f1a; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;"
            ),
            style="margin-bottom: 12px;"
        ) if check_options else None,
        style="background: #2d2d1f; padding: 12px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #f59e0b;"
    )


def ErrorFilterPanel(filters: dict, current_model: str = None, current_use_case: str = None, error_type: str = None):
    """Filter panel for the errors page."""
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
                    *[Option(
                        f"{uc['name']} ({uc['count']})" if uc.get('name') else f"{uc['hash'][:8]}... ({uc['count']})",
                        value=uc['hash'],
                        selected=uc['hash'] == current_use_case
                    ) for uc in filters.get('use_cases', [])],
                    name="use_case",
                    style="width: 100%; padding: 8px; background: #0f0f1a; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;"
                ),
                style="margin-bottom: 12px;"
            ),
            Div(
                Label("Error Type:", style="color: #94a3b8; font-size: 0.85em;"),
                Select(
                    Option("All Errors", value="", selected=not error_type),
                    Option("User Disagreements", value="user_disagreement", selected=error_type == "user_disagreement"),
                    Option("Empty Response", value="empty_response", selected=error_type == "empty_response"),
                    Option("Refusal", value="refusal", selected=error_type == "refusal"),
                    Option("Error Status", value="error_status", selected=error_type == "error_status"),
                    name="error_type",
                    style="width: 100%; padding: 8px; background: #0f0f1a; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;"
                ),
                style="margin-bottom: 12px;"
            ),
            Button("Apply Filters", type="submit", style="width: 100%; padding: 10px; background: #3b82f6; color: white; border: none; border-radius: 6px; cursor: pointer;"),
            A("Clear Filters", href="/errors", style="display: block; text-align: center; margin-top: 8px; color: #64748b; font-size: 0.85em;"),
            method="get",
            action="/errors"
        ),
        style="background: #1f2940; padding: 16px; border-radius: 12px; border: 1px solid #2d3748; margin-bottom: 16px;"
    )
