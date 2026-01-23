import json
import sqlite3
import logging
import time
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

from config import TRACES_DB, PROMPTS_DIR, ANALYSIS_CACHE
from metadata_loader import (
    get_trace_metadata, get_metadata_for_trace_id, get_available_workspaces,
    get_metadata_stats, metadata_table_exists
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    total_elements: int  # Total trace elements (rows in DB)
    unique_trace_ids: int  # Unique trace IDs
    unique_use_cases: int
    time_period_start: str
    time_period_end: str
    time_span_days: float
    model_distribution: dict
    use_case_distribution: dict
    multi_turn_stats: dict
    tool_call_stats: dict
    metadata_stats: dict
    error_detection_stats: dict
    cost_stats: dict
    token_stats: dict
    response_time_stats: dict
    multi_element_stats: dict  # Stats about traces with multiple elements


def get_db_connection():
    return sqlite3.connect(TRACES_DB)


def run_analysis() -> AnalysisResult:
    """Run comprehensive analysis on processed traces."""
    start_time = time.time()
    logger.debug("Starting analysis...")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Total elements (rows)
    cursor.execute('SELECT COUNT(*) FROM traces')
    total_elements = cursor.fetchone()[0]
    logger.debug(f"Total elements in database: {total_elements:,}")

    # Unique trace IDs
    cursor.execute('SELECT COUNT(DISTINCT trace_id) FROM traces')
    unique_trace_ids = cursor.fetchone()[0]

    # Unique use cases (system prompts)
    cursor.execute('SELECT COUNT(DISTINCT system_prompt_hash) FROM traces')
    unique_use_cases = cursor.fetchone()[0]

    # Multi-element trace stats
    cursor.execute('''
        SELECT trace_id, COUNT(*) as element_count
        FROM traces
        GROUP BY trace_id
        HAVING COUNT(*) > 1
    ''')
    multi_element_traces = cursor.fetchall()
    traces_with_multiple_elements = len(multi_element_traces)
    max_elements_per_trace = max([r[1] for r in multi_element_traces]) if multi_element_traces else 1
    avg_elements_per_trace = sum([r[1] for r in multi_element_traces]) / len(multi_element_traces) if multi_element_traces else 1

    multi_element_stats = {
        'traces_with_multiple_elements': traces_with_multiple_elements,
        'max_elements_per_trace': max_elements_per_trace,
        'avg_elements_per_trace': round(avg_elements_per_trace, 2),
        'total_extra_elements': total_elements - unique_trace_ids
    }

    # Time period
    cursor.execute('''
        SELECT MIN(parsed_timestamp), MAX(parsed_timestamp)
        FROM traces WHERE parsed_timestamp IS NOT NULL
    ''')
    time_row = cursor.fetchone()
    time_start = time_row[0] or 'Unknown'
    time_end = time_row[1] or 'Unknown'

    time_span_days = 0.0
    if time_start != 'Unknown' and time_end != 'Unknown':
        try:
            start_dt = datetime.fromisoformat(time_start)
            end_dt = datetime.fromisoformat(time_end)
            time_span_days = (end_dt - start_dt).total_seconds() / 86400
        except:
            pass

    # Model distribution
    cursor.execute('''
        SELECT model, COUNT(*) as count FROM traces
        GROUP BY model ORDER BY count DESC
    ''')
    model_distribution = {row[0]: row[1] for row in cursor.fetchall()}

    # Use case distribution with prompt info and use_case_name from prompts table
    cursor.execute('''
        SELECT t.system_prompt_hash, t.system_prompt_file, COUNT(*) as count,
               t.model, p.use_case_name, p.use_case_version, p.use_case_type, p.workspace, p.model_config
        FROM traces t
        LEFT JOIN prompts p ON t.system_prompt_hash = p.prompt_hash
        GROUP BY t.system_prompt_hash
        ORDER BY count DESC
    ''')
    use_case_rows = cursor.fetchall()
    use_case_distribution = {}
    for row in use_case_rows:
        prompt_hash, filename, count, sample_model, use_case_name, use_case_version, use_case_type, workspace, model_config = row

        # Use the matched use_case_name if available, otherwise extract from prompt file
        if use_case_name:
            description = use_case_name
            if use_case_version:
                description = f"{use_case_name} ({use_case_version})"
        else:
            # Fallback: Load first 200 chars of prompt for description
            prompt_path = PROMPTS_DIR / filename if filename else None
            description = ""
            if prompt_path and prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    content = f.read()
                    # Extract a meaningful title from the prompt
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('<') and not line.startswith('#'):
                            description = line[:100]
                            break
                    if not description:
                        description = content[:100].replace('\n', ' ')

        use_case_distribution[prompt_hash] = {
            'filename': filename,
            'count': count,
            'description': description,
            'sample_model': sample_model,
            'use_case_name': use_case_name,
            'use_case_version': use_case_version,
            'use_case_type': use_case_type,
            'workspace': workspace,
            'model_config': model_config
        }

    # Multi-turn stats
    cursor.execute('''
        SELECT
            SUM(CASE WHEN is_multi_turn = 1 THEN 1 ELSE 0 END) as multi_turn,
            SUM(CASE WHEN is_multi_turn = 0 THEN 1 ELSE 0 END) as single_turn,
            AVG(num_turns) as avg_turns,
            MAX(num_turns) as max_turns
        FROM traces
    ''')
    mt_row = cursor.fetchone()
    multi_turn_stats = {
        'multi_turn_count': mt_row[0] or 0,
        'single_turn_count': mt_row[1] or 0,
        'avg_turns': round(mt_row[2] or 0, 2),
        'max_turns': mt_row[3] or 0
    }

    # Tool call stats
    cursor.execute('''
        SELECT
            SUM(CASE WHEN has_tool_calls = 1 THEN 1 ELSE 0 END) as with_tools,
            SUM(CASE WHEN has_tool_calls = 0 THEN 1 ELSE 0 END) as without_tools
        FROM traces
    ''')
    tc_row = cursor.fetchone()
    tool_call_stats = {
        'with_tool_calls': tc_row[0] or 0,
        'without_tool_calls': tc_row[1] or 0
    }

    # Metadata stats
    cursor.execute('''
        SELECT
            SUM(CASE WHEN has_metadata = 1 THEN 1 ELSE 0 END) as with_metadata,
            SUM(CASE WHEN has_metadata = 0 THEN 1 ELSE 0 END) as without_metadata
        FROM traces
    ''')
    meta_row = cursor.fetchone()

    # Get unique metadata keys
    cursor.execute('SELECT metadata_keys FROM traces WHERE metadata_keys != ""')
    all_meta_keys = []
    for row in cursor.fetchall():
        all_meta_keys.extend(row[0].split(','))
    meta_key_counts = Counter(all_meta_keys)

    metadata_stats = {
        'with_metadata': meta_row[0] or 0,
        'without_metadata': meta_row[1] or 0,
        'unique_metadata_keys': dict(meta_key_counts.most_common(20))
    }

    # Error detection stats
    cursor.execute('''
        SELECT
            SUM(CASE WHEN user_disagreement_detected = 1 THEN 1 ELSE 0 END) as user_disagreements,
            SUM(CASE WHEN potential_error_indicators != '' THEN 1 ELSE 0 END) as with_error_indicators
        FROM traces
    ''')
    err_row = cursor.fetchone()

    # Get error indicator breakdown
    cursor.execute('SELECT potential_error_indicators FROM traces WHERE potential_error_indicators != ""')
    all_indicators = []
    for row in cursor.fetchall():
        all_indicators.extend(row[0].split(','))
    indicator_counts = Counter(all_indicators)

    error_detection_stats = {
        'user_disagreements': err_row[0] or 0,
        'traces_with_error_indicators': err_row[1] or 0,
        'error_indicator_breakdown': dict(indicator_counts.most_common(20))
    }

    # Cost stats
    cursor.execute('''
        SELECT
            SUM(cost) as total_cost,
            AVG(cost) as avg_cost,
            MIN(cost) as min_cost,
            MAX(cost) as max_cost
        FROM traces
    ''')
    cost_row = cursor.fetchone()
    cost_stats = {
        'total_cost': round(cost_row[0] or 0, 4),
        'avg_cost': round(cost_row[1] or 0, 6),
        'min_cost': round(cost_row[2] or 0, 6),
        'max_cost': round(cost_row[3] or 0, 4)
    }

    # Token stats
    cursor.execute('''
        SELECT
            SUM(total_tokens) as total_tokens,
            AVG(total_tokens) as avg_tokens,
            AVG(prompt_tokens) as avg_prompt_tokens,
            AVG(completion_tokens) as avg_completion_tokens
        FROM traces
    ''')
    token_row = cursor.fetchone()
    token_stats = {
        'total_tokens': token_row[0] or 0,
        'avg_tokens': round(token_row[1] or 0, 0),
        'avg_prompt_tokens': round(token_row[2] or 0, 0),
        'avg_completion_tokens': round(token_row[3] or 0, 0)
    }

    # Response time stats
    cursor.execute('''
        SELECT
            AVG(response_time) as avg_response_time,
            MIN(response_time) as min_response_time,
            MAX(response_time) as max_response_time
        FROM traces WHERE response_time > 0
    ''')
    rt_row = cursor.fetchone()
    response_time_stats = {
        'avg_response_time_ms': round(rt_row[0] or 0, 0),
        'min_response_time_ms': rt_row[1] or 0,
        'max_response_time_ms': rt_row[2] or 0
    }

    conn.close()

    result = AnalysisResult(
        total_elements=total_elements,
        unique_trace_ids=unique_trace_ids,
        unique_use_cases=unique_use_cases,
        time_period_start=time_start,
        time_period_end=time_end,
        time_span_days=round(time_span_days, 2),
        model_distribution=model_distribution,
        use_case_distribution=use_case_distribution,
        multi_turn_stats=multi_turn_stats,
        tool_call_stats=tool_call_stats,
        metadata_stats=metadata_stats,
        error_detection_stats=error_detection_stats,
        cost_stats=cost_stats,
        token_stats=token_stats,
        response_time_stats=response_time_stats,
        multi_element_stats=multi_element_stats
    )

    # Cache the result
    with open(ANALYSIS_CACHE, 'w') as f:
        json.dump(asdict(result), f, indent=2)

    elapsed = time.time() - start_time
    logger.debug(f"Analysis completed in {elapsed:.3f}s")

    return result


def get_random_traces_for_use_case(prompt_hash: str, limit: int = 5) -> list[dict]:
    """Get random traces for a specific use case."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if metadata table exists
    has_metadata_table = False
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trace_metadata'")
        has_metadata_table = cursor.fetchone() is not None
    except:
        pass

    if has_metadata_table:
        cursor.execute('''
            SELECT t.element_id, t.trace_id, t.model, t.num_turns, t.has_tool_calls,
                   t.cost, t.total_tokens, t.response_time, t.created_at,
                   t.potential_error_indicators, t.user_disagreement_detected,
                   (SELECT COUNT(*) FROM traces t2 WHERE t2.trace_id = t.trace_id) as element_count,
                   m.workspace_name, m.patient_id, m.patient_name
            FROM traces t
            LEFT JOIN trace_metadata m ON t.element_id = m.element_id
            WHERE t.system_prompt_hash = ?
            ORDER BY RANDOM()
            LIMIT ?
        ''', (prompt_hash, limit))
    else:
        cursor.execute('''
            SELECT t.element_id, t.trace_id, t.model, t.num_turns, t.has_tool_calls,
                   t.cost, t.total_tokens, t.response_time, t.created_at,
                   t.potential_error_indicators, t.user_disagreement_detected,
                   (SELECT COUNT(*) FROM traces t2 WHERE t2.trace_id = t.trace_id) as element_count
            FROM traces t
            WHERE t.system_prompt_hash = ?
            ORDER BY RANDOM()
            LIMIT ?
        ''', (prompt_hash, limit))

    traces = []
    for row in cursor.fetchall():
        trace_dict = {
            'element_id': row[0],
            'trace_id': row[1],
            'model': row[2],
            'num_turns': row[3],
            'has_tool_calls': bool(row[4]),
            'cost': row[5],
            'total_tokens': row[6],
            'response_time': row[7],
            'created_at': row[8],
            'potential_error_indicators': row[9],
            'user_disagreement_detected': bool(row[10]),
            'element_count': row[11]
        }
        # Add metadata fields if available
        if has_metadata_table and len(row) > 12:
            trace_dict['workspace_name'] = row[12]
            trace_dict['patient_id'] = row[13]
            trace_dict['patient_name'] = row[14]
        traces.append(trace_dict)

    conn.close()
    return traces


def get_trace_detail(trace_id: str) -> Optional[dict]:
    """Get full trace detail including all elements for this trace_id."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all elements for this trace_id, ordered by timestamp
    cursor.execute('''
        SELECT * FROM traces
        WHERE trace_id = ?
        ORDER BY parsed_timestamp ASC
    ''', (trace_id,))
    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return None

    columns = [desc[0] for desc in cursor.description]

    # Build list of all elements
    elements = []
    for row in rows:
        element_info = dict(zip(columns, row))

        # Get raw data for this element
        cursor.execute('SELECT data FROM raw_traces WHERE element_id = ?', (element_info['element_id'],))
        raw_row = cursor.fetchone()
        if raw_row:
            element_info['raw_data'] = json.loads(raw_row[0])

        # Get metadata for this element (patient/workspace info)
        metadata = get_trace_metadata(element_info['element_id'])
        if metadata:
            element_info['patient_metadata'] = metadata

        elements.append(element_info)

    conn.close()

    # Get first element's metadata for summary
    first_metadata = elements[0].get('patient_metadata') if elements else None

    # Return combined info
    return {
        'trace_id': trace_id,
        'element_count': len(elements),
        'elements': elements,
        # Use first element for summary info
        'model': elements[0].get('model'),
        'system_prompt_hash': elements[0].get('system_prompt_hash'),
        'system_prompt_file': elements[0].get('system_prompt_file'),
        # Patient/workspace info from metadata
        'patient_id': first_metadata.get('patient_id') if first_metadata else None,
        'patient_name': first_metadata.get('patient_name') if first_metadata else None,
        'workspace_name': first_metadata.get('workspace_name') if first_metadata else None,
        'workspace_id': first_metadata.get('workspace_id') if first_metadata else None,
    }


def get_element_detail(element_id: str) -> Optional[dict]:
    """Get detail for a specific element."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM traces WHERE element_id = ?', (element_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    columns = [desc[0] for desc in cursor.description]
    element_info = dict(zip(columns, row))

    # Get raw data
    cursor.execute('SELECT data FROM raw_traces WHERE element_id = ?', (element_id,))
    raw_row = cursor.fetchone()
    if raw_row:
        element_info['raw_data'] = json.loads(raw_row[0])

    # Get element count for this trace
    cursor.execute('SELECT COUNT(*) FROM traces WHERE trace_id = ?', (element_info['trace_id'],))
    element_info['total_elements'] = cursor.fetchone()[0]

    conn.close()

    # Get metadata for this element (patient/workspace info)
    metadata = get_trace_metadata(element_id)
    if metadata:
        element_info['patient_metadata'] = metadata

    return element_info


def get_system_prompt_content(prompt_hash: str) -> str:
    """Get system prompt content by hash."""
    prompt_file = PROMPTS_DIR / f"prompt_{prompt_hash}.txt"
    if prompt_file.exists():
        with open(prompt_file, 'r') as f:
            return f.read()
    return "Prompt not found"


def get_use_case_info(prompt_hash: str) -> Optional[dict]:
    """Get use case info including name, version, type, workspace, model_config by prompt hash."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT prompt_hash, use_case_name, use_case_version, use_case_type, workspace, trace_count, model_config
        FROM prompts
        WHERE prompt_hash = ?
    ''', (prompt_hash,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'prompt_hash': row[0],
            'use_case_name': row[1],
            'use_case_version': row[2],
            'use_case_type': row[3],
            'workspace': row[4],
            'trace_count': row[5],
            'model_config': row[6]
        }
    return None


def get_traces_with_errors(
    limit: int = 50,
    model: str = None,
    use_case: str = None,
    error_type: str = None
) -> list[dict]:
    """Get traces that have potential error indicators."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Build WHERE clause
    conditions = ["(potential_error_indicators != '' OR user_disagreement_detected = 1)"]
    params = []

    if model:
        conditions.append('model = ?')
        params.append(model)

    if use_case:
        conditions.append('system_prompt_hash = ?')
        params.append(use_case)

    if error_type:
        if error_type == 'user_disagreement':
            conditions.append('user_disagreement_detected = 1')
        elif error_type == 'empty_response':
            conditions.append("potential_error_indicators LIKE '%empty_response%'")
        elif error_type == 'refusal':
            conditions.append("potential_error_indicators LIKE '%refusal%'")
        elif error_type == 'error_status':
            conditions.append("potential_error_indicators LIKE '%error_status%'")

    where_clause = f"WHERE {' AND '.join(conditions)}"

    sql = f'''
        SELECT element_id, trace_id, model, system_prompt_hash, num_turns,
               potential_error_indicators, user_disagreement_detected, created_at,
               (SELECT COUNT(*) FROM traces t2 WHERE t2.trace_id = traces.trace_id) as element_count
        FROM traces
        {where_clause}
        ORDER BY parsed_timestamp DESC
        LIMIT ?
    '''

    cursor.execute(sql, params + [limit])

    traces = []
    for row in cursor.fetchall():
        traces.append({
            'element_id': row[0],
            'trace_id': row[1],
            'model': row[2],
            'system_prompt_hash': row[3],
            'num_turns': row[4],
            'potential_error_indicators': row[5],
            'user_disagreement_detected': bool(row[6]),
            'created_at': row[7],
            'element_count': row[8]
        })

    conn.close()
    return traces


def get_traces_with_multiple_elements(limit: int = 50) -> list[dict]:
    """Get traces that have multiple elements."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT trace_id, COUNT(*) as element_count,
               MIN(created_at) as first_created,
               MAX(created_at) as last_created,
               GROUP_CONCAT(DISTINCT model) as models
        FROM traces
        GROUP BY trace_id
        HAVING COUNT(*) > 1
        ORDER BY element_count DESC
        LIMIT ?
    ''', (limit,))

    traces = []
    for row in cursor.fetchall():
        traces.append({
            'trace_id': row[0],
            'element_count': row[1],
            'first_created': row[2],
            'last_created': row[3],
            'models': row[4]
        })

    conn.close()
    return traces


def get_filtered_traces(
    page: int = 1,
    limit: int = 50,
    model: str = None,
    use_case: str = None,
    has_errors: bool = None,
    multi_element_only: bool = False,
    with_metadata: bool = None,
    multi_turn: bool = None,
    workspace: str = None
) -> tuple[list[dict], int]:
    """Get filtered list of traces with pagination."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if metadata table exists for workspace filtering
    has_metadata_table = False
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trace_metadata'")
        has_metadata_table = cursor.fetchone() is not None
    except:
        pass

    # Build WHERE clause
    conditions = []
    params = []
    join_clause = ""

    if model:
        conditions.append('t.model = ?')
        params.append(model)

    if use_case:
        conditions.append('t.system_prompt_hash = ?')
        params.append(use_case)

    if has_errors:
        conditions.append("(t.potential_error_indicators != '' OR t.user_disagreement_detected = 1)")

    if with_metadata:
        conditions.append('t.has_metadata = 1')

    if multi_turn:
        conditions.append('t.is_multi_turn = 1')

    # Workspace filter requires join with trace_metadata table
    if workspace and has_metadata_table:
        join_clause = "LEFT JOIN trace_metadata m ON t.element_id = m.element_id"
        conditions.append('m.workspace_name = ?')
        params.append(workspace)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Get total count
    if multi_element_only:
        count_sql = f'''
            SELECT COUNT(DISTINCT t.trace_id) FROM traces t {join_clause} {where_clause}
            {'AND' if where_clause else 'WHERE'} t.trace_id IN (
                SELECT trace_id FROM traces GROUP BY trace_id HAVING COUNT(*) > 1
            )
        '''
    else:
        count_sql = f'SELECT COUNT(*) FROM traces t {join_clause} {where_clause}'

    cursor.execute(count_sql, params)
    total = cursor.fetchone()[0]

    # Get traces
    offset = (page - 1) * limit

    # Base select columns
    select_cols = '''
        t.element_id, t.trace_id, t.model, t.num_turns, t.has_tool_calls, t.cost,
        t.total_tokens, t.response_time, t.created_at, t.potential_error_indicators,
        t.user_disagreement_detected, t.system_prompt_hash,
        (SELECT COUNT(*) FROM traces t2 WHERE t2.trace_id = t.trace_id) as element_count
    '''

    # Add metadata columns if table exists
    if has_metadata_table:
        select_cols += ', m.workspace_name, m.patient_id, m.patient_name'
        if not join_clause:
            join_clause = "LEFT JOIN trace_metadata m ON t.element_id = m.element_id"

    if multi_element_only:
        sql = f'''
            SELECT {select_cols}
            FROM traces t
            {join_clause}
            {where_clause}
            {'AND' if where_clause else 'WHERE'} t.trace_id IN (
                SELECT trace_id FROM traces GROUP BY trace_id HAVING COUNT(*) > 1
            )
            ORDER BY t.parsed_timestamp DESC
            LIMIT ? OFFSET ?
        '''
    else:
        sql = f'''
            SELECT {select_cols}
            FROM traces t
            {join_clause}
            {where_clause}
            ORDER BY t.parsed_timestamp DESC
            LIMIT ? OFFSET ?
        '''

    cursor.execute(sql, params + [limit, offset])

    traces = []
    for row in cursor.fetchall():
        trace_dict = {
            'element_id': row[0],
            'trace_id': row[1],
            'model': row[2],
            'num_turns': row[3],
            'has_tool_calls': bool(row[4]),
            'cost': row[5],
            'total_tokens': row[6],
            'response_time': row[7],
            'created_at': row[8],
            'potential_error_indicators': row[9],
            'user_disagreement_detected': bool(row[10]),
            'system_prompt_hash': row[11],
            'element_count': row[12]
        }
        # Add metadata fields if available
        if has_metadata_table and len(row) > 13:
            trace_dict['workspace_name'] = row[13]
            trace_dict['patient_id'] = row[14]
            trace_dict['patient_name'] = row[15]
        traces.append(trace_dict)

    conn.close()
    return traces, total


def get_available_filters() -> dict:
    """Get available filter options."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get unique models
    cursor.execute('SELECT DISTINCT model FROM traces ORDER BY model')
    models = [row[0] for row in cursor.fetchall()]

    # Get unique use cases with counts
    cursor.execute('''
        SELECT system_prompt_hash, COUNT(*) as count
        FROM traces
        GROUP BY system_prompt_hash
        ORDER BY count DESC
    ''')
    use_cases = [{'hash': row[0], 'count': row[1]} for row in cursor.fetchall()]

    conn.close()

    # Get workspaces from metadata table
    workspaces = get_available_workspaces()

    return {
        'models': models,
        'use_cases': use_cases,
        'workspaces': workspaces
    }


def get_model_stats_by_use_case() -> dict:
    """Get model usage breakdown by use case."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Normalize model names by stripping provider prefix (e.g., "google/gemini-2.5-flash" -> "gemini-2.5-flash")
    cursor.execute('''
        SELECT system_prompt_hash,
               CASE WHEN INSTR(model, '/') > 0
                    THEN SUBSTR(model, INSTR(model, '/') + 1)
                    ELSE model
               END as normalized_model,
               COUNT(*) as count,
               AVG(cost) as avg_cost,
               AVG(response_time) as avg_response_time
        FROM traces
        GROUP BY system_prompt_hash, normalized_model
        ORDER BY system_prompt_hash, count DESC
    ''')

    result = {}
    for row in cursor.fetchall():
        prompt_hash = row[0]
        if prompt_hash not in result:
            result[prompt_hash] = []
        result[prompt_hash].append({
            'model': row[1],
            'count': row[2],
            'avg_cost': round(row[3] or 0, 6),
            'avg_response_time': round(row[4] or 0, 0)
        })

    conn.close()
    return result


def get_workspace_stats_by_use_case() -> dict:
    """
    Get workspace stats from metadata table grouped by use case (system_prompt_hash).

    Returns:
        dict: {prompt_hash: [{'workspace': str, 'region': str, 'count': int}, ...]}
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if metadata table exists
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trace_metadata'")
        if not cursor.fetchone():
            conn.close()
            return {}
    except:
        conn.close()
        return {}

    # Query workspace stats joined with traces to get system_prompt_hash
    cursor.execute('''
        SELECT t.system_prompt_hash, m.workspace_name, m.replica_source, COUNT(*) as count
        FROM traces t
        JOIN trace_metadata m ON t.element_id = m.element_id
        WHERE m.workspace_name IS NOT NULL AND m.workspace_name != ''
        GROUP BY t.system_prompt_hash, m.workspace_name, m.replica_source
        ORDER BY t.system_prompt_hash, count DESC
    ''')

    result = {}
    for row in cursor.fetchall():
        prompt_hash, workspace, replica, count = row

        # Extract region from replica_source (e.g., "lcusreplica" -> "US")
        region = "?"
        if replica:
            replica_lower = replica.lower()
            if 'lcus' in replica_lower:
                region = "US"
            elif 'lceu' in replica_lower:
                region = "EU"
            elif 'lcin' in replica_lower:
                region = "IN"
            elif 'us' in replica_lower:
                region = "US"
            elif 'eu' in replica_lower:
                region = "EU"
            elif 'in' in replica_lower:
                region = "IN"

        if prompt_hash not in result:
            result[prompt_hash] = []

        result[prompt_hash].append({
            'workspace': workspace,
            'region': region,
            'count': count
        })

    conn.close()
    return result


if __name__ == "__main__":
    result = run_analysis()
    print("\n=== ANALYSIS RESULTS ===")
    print(f"Total elements: {result.total_elements}")
    print(f"Unique trace IDs: {result.unique_trace_ids}")
    print(f"Unique use cases: {result.unique_use_cases}")
    print(f"Multi-element stats: {result.multi_element_stats}")
    print(f"Time period: {result.time_period_start} to {result.time_period_end}")
    print(f"Models: {result.model_distribution}")
    print(f"Cost stats: {result.cost_stats}")
    print(f"Error detection: {result.error_detection_stats}")
