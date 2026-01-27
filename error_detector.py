"""
Error Detection Module for TraceAudit

Implements two-level error detection for traces with patient metadata:
- Level 1: Fast, rule-based checks (no LLM)
- Level 2: LLM-based checks using Gemini Flash via Portkey

Supported use cases:
- worklist_rad_report_data_extraction
- worklist_suggest_next_steps
- worklist_patient_summary
"""

import json
import logging
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

import yaml

from config import TRACES_DB, ERROR_DETECTION_PROMPTS
from metadata_loader import get_trace_metadata

logger = logging.getLogger(__name__)

# Use case to prompt hash mapping
USE_CASE_HASHES = {
    '61422e2ac4ee4efb': 'worklist_rad_report_data_extraction',
    '9a3acb64787e8a9f': 'worklist_rad_report_data_extraction',
    '995558dcc87664fa': 'worklist_suggest_next_steps',
    'aec9115a1ea76ab0': 'worklist_patient_summary',
}

# Reverse mapping
HASH_TO_USE_CASE = USE_CASE_HASHES


def get_db_connection():
    return sqlite3.connect(TRACES_DB)


def init_error_detection_table(conn: sqlite3.Connection = None):
    """Initialize the trace_errors table for storing error detection results."""
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trace_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            element_id TEXT NOT NULL,
            check_id TEXT NOT NULL,
            check_level INTEGER NOT NULL,
            has_error INTEGER NOT NULL,
            error_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (element_id) REFERENCES traces(element_id),
            UNIQUE(element_id, check_id)
        )
    ''')

    # Create indexes for efficient queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_element_id ON trace_errors(element_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_has_error ON trace_errors(has_error)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_check_id ON trace_errors(check_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_errors_check_level ON trace_errors(check_level)')

    conn.commit()
    logger.info("trace_errors table initialized")

    if should_close:
        conn.close()


def extract_output_from_trace(raw_data: dict) -> tuple[str, Optional[dict]]:
    """
    Extract the LLM output from a trace.

    Returns:
        tuple: (output_type, output_data)
        - output_type: 'tool_call', 'content', or 'empty'
        - output_data: parsed JSON for tool calls, string for content
    """
    response = raw_data.get('response', {})
    choices = response.get('choices', [])

    if not choices:
        return 'empty', None

    message = choices[0].get('message', {})

    # Check for tool calls (rad_report and suggest_next_steps use this)
    tool_calls = message.get('tool_calls', [])
    if tool_calls:
        for tc in tool_calls:
            func = tc.get('function', {})
            if func.get('name') == 'final_result':
                try:
                    args = json.loads(func.get('arguments', '{}'))
                    return 'tool_call', args
                except json.JSONDecodeError:
                    return 'tool_call_parse_error', func.get('arguments')

    # Direct content (patient_summary uses this)
    content = message.get('content', '')
    if content:
        return 'content', content

    return 'empty', None


def extract_input_from_trace(raw_data: dict) -> str:
    """Extract the user input/prompt from a trace."""
    request = raw_data.get('request', {})
    messages = request.get('messages', [])

    # Get the last user message
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            return msg.get('content', '')

    return ''


def parse_available_tags(messages: list) -> list[dict]:
    """Parse available_tags from the system prompt in messages."""
    for msg in messages:
        if msg.get('role') == 'system':
            content = msg.get('content', '')
            # Look for available_tags in the prompt
            if 'available_tags' in content.lower():
                # Try to extract JSON array
                try:
                    match = re.search(r'available_tags["\s:]+(\[.*?\])', content, re.DOTALL | re.IGNORECASE)
                    if match:
                        return json.loads(match.group(1))
                except:
                    pass
    return []


def parse_available_forms(messages: list) -> list[dict]:
    """Parse available_forms from the system prompt in messages."""
    for msg in messages:
        if msg.get('role') == 'system':
            content = msg.get('content', '')
            if 'available_forms' in content.lower():
                try:
                    match = re.search(r'available_forms["\s:]+(\[.*?\])', content, re.DOTALL | re.IGNORECASE)
                    if match:
                        return json.loads(match.group(1))
                except:
                    pass
    return []


def parse_available_letters(messages: list) -> list[dict]:
    """Parse available_letters from the system prompt in messages."""
    for msg in messages:
        if msg.get('role') == 'system':
            content = msg.get('content', '')
            if 'available_letters' in content.lower():
                try:
                    match = re.search(r'available_letters["\s:]+(\[.*?\])', content, re.DOTALL | re.IGNORECASE)
                    if match:
                        return json.loads(match.group(1))
                except:
                    pass
    return []


# Level 1 Checks

def check_l1_empty_response(output_type: str, output_data) -> dict:
    """L1: Check for empty response."""
    has_error = output_type == 'empty' or output_data is None
    return {
        'check_id': 'L1_EMPTY_RESPONSE',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': 'Response is empty - no content or tool calls' if has_error else ''
    }


def check_l1_json_parse_error(output_type: str, output_data) -> dict:
    """L1: Check for JSON parse errors in tool call arguments."""
    has_error = output_type == 'tool_call_parse_error'
    return {
        'check_id': 'L1_JSON_PARSE_ERROR',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': 'Failed to parse tool call arguments as JSON' if has_error else ''
    }


def check_l1_refusal(output_type: str, output_data) -> dict:
    """L1: Check for refusal patterns in response."""
    refusal_patterns = ['i cannot', "i can't", "i'm unable", 'i apologize', 'i am unable']

    content = ''
    if output_type == 'content' and isinstance(output_data, str):
        content = output_data.lower()
    elif output_type == 'tool_call' and isinstance(output_data, dict):
        content = json.dumps(output_data).lower()

    has_error = any(pattern in content for pattern in refusal_patterns)
    matched = [p for p in refusal_patterns if p in content]

    return {
        'check_id': 'L1_REFUSAL',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': f'Response contains refusal pattern: {matched[0]}' if has_error else ''
    }


def check_l1_patient_mismatch(output_data, metadata: dict) -> dict:
    """L1: Check if patient ID in output differs from metadata."""
    if not metadata or not metadata.get('patient_id'):
        return {
            'check_id': 'L1_PATIENT_MISMATCH',
            'check_level': 1,
            'has_error': False,
            'error_reason': ''
        }

    expected_patient_id = str(metadata.get('patient_id', ''))
    output_str = json.dumps(output_data) if isinstance(output_data, dict) else str(output_data)

    # Look for patient_id in output that doesn't match
    patient_id_match = re.search(r'patient_id["\s:]+["\']?(\d+)', output_str, re.IGNORECASE)
    if patient_id_match:
        found_id = patient_id_match.group(1)
        if found_id != expected_patient_id:
            return {
                'check_id': 'L1_PATIENT_MISMATCH',
                'check_level': 1,
                'has_error': True,
                'error_reason': f'Patient ID mismatch: expected {expected_patient_id}, found {found_id}'
            }

    return {
        'check_id': 'L1_PATIENT_MISMATCH',
        'check_level': 1,
        'has_error': False,
        'error_reason': ''
    }


# rad_report_data_extraction specific checks

def check_l1_missing_heading(output_data: dict) -> dict:
    """L1: Check for missing heading in rad_report output."""
    heading = output_data.get('heading') if isinstance(output_data, dict) else None
    has_error = not heading or heading.strip() == ''
    return {
        'check_id': 'L1_MISSING_HEADING',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': 'Missing or empty heading field' if has_error else ''
    }


def check_l1_missing_highlight(output_data: dict) -> dict:
    """L1: Check for missing highlighted_report in rad_report output."""
    highlighted = output_data.get('highlighted_report') if isinstance(output_data, dict) else None
    has_error = not highlighted or highlighted.strip() == ''
    return {
        'check_id': 'L1_MISSING_HIGHLIGHT',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': 'Missing or empty highlighted_report field' if has_error else ''
    }


def check_l1_unbalanced_highlight_tags(output_data: dict) -> dict:
    """L1: Check for unbalanced highlight tags in rad_report output."""
    highlighted = output_data.get('highlighted_report', '') if isinstance(output_data, dict) else ''

    open_count = highlighted.count('<highlighted_text>')
    close_count = highlighted.count('</highlighted_text>')

    has_error = open_count != close_count
    return {
        'check_id': 'L1_UNBALANCED_HIGHLIGHT',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': f'Unbalanced highlight tags: {open_count} opening, {close_count} closing' if has_error else ''
    }


def check_l1_invalid_tag_id(output_data: dict, available_tags: list) -> dict:
    """L1: Check if tag IDs in output are valid."""
    if not isinstance(output_data, dict) or not available_tags:
        return {
            'check_id': 'L1_INVALID_TAG_ID',
            'check_level': 1,
            'has_error': False,
            'error_reason': ''
        }

    tags = output_data.get('tags', [])
    valid_tag_ids = {str(t.get('id', '')) for t in available_tags if isinstance(t, dict)}

    invalid_tags = []
    for tag in tags:
        if isinstance(tag, dict):
            tag_id = str(tag.get('id', ''))
            if tag_id and tag_id not in valid_tag_ids:
                invalid_tags.append(tag_id)

    has_error = len(invalid_tags) > 0
    return {
        'check_id': 'L1_INVALID_TAG_ID',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': f'Invalid tag IDs: {invalid_tags}' if has_error else ''
    }


# suggest_next_steps specific checks

def check_l1_invalid_action_type(output_data: dict) -> dict:
    """L1: Check for invalid action types in suggest_next_steps."""
    valid_types = {'reminder', 'form', 'letter', 'tags'}

    if not isinstance(output_data, dict):
        return {
            'check_id': 'L1_INVALID_ACTION_TYPE',
            'check_level': 1,
            'has_error': False,
            'error_reason': ''
        }

    next_steps = output_data.get('next_steps', [])
    invalid_types = []

    for step in next_steps:
        if isinstance(step, dict):
            step_type = step.get('type', '')
            if step_type and step_type.lower() not in valid_types:
                invalid_types.append(step_type)

    has_error = len(invalid_types) > 0
    return {
        'check_id': 'L1_INVALID_ACTION_TYPE',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': f'Invalid action types: {invalid_types}' if has_error else ''
    }


def check_l1_invalid_date(output_data: dict) -> dict:
    """L1: Check for invalid date format in remind_after fields."""
    if not isinstance(output_data, dict):
        return {
            'check_id': 'L1_INVALID_DATE',
            'check_level': 1,
            'has_error': False,
            'error_reason': ''
        }

    next_steps = output_data.get('next_steps', [])
    invalid_dates = []

    for step in next_steps:
        if isinstance(step, dict) and step.get('type') == 'reminder':
            action = step.get('action', {})
            remind_after = action.get('remind_after', '') if isinstance(action, dict) else ''
            if remind_after:
                try:
                    datetime.fromisoformat(remind_after.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    invalid_dates.append(remind_after)

    has_error = len(invalid_dates) > 0
    return {
        'check_id': 'L1_INVALID_DATE',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': f'Invalid date formats: {invalid_dates}' if has_error else ''
    }


def check_l1_missing_action_field(output_data: dict) -> dict:
    """L1: Check for missing required fields in actions."""
    if not isinstance(output_data, dict):
        return {
            'check_id': 'L1_MISSING_ACTION_FIELD',
            'check_level': 1,
            'has_error': False,
            'error_reason': ''
        }

    next_steps = output_data.get('next_steps', [])
    missing = []

    for i, step in enumerate(next_steps):
        if not isinstance(step, dict):
            continue

        step_type = step.get('type', '')
        action = step.get('action', {})

        if step_type == 'reminder':
            if not action.get('context'):
                missing.append(f'Step {i+1}: reminder missing context')
        elif step_type == 'form':
            if not action.get('id'):
                missing.append(f'Step {i+1}: form missing id')

    has_error = len(missing) > 0
    return {
        'check_id': 'L1_MISSING_ACTION_FIELD',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': '; '.join(missing) if has_error else ''
    }


# patient_summary specific checks

def check_l1_too_short(output_data: str) -> dict:
    """L1: Check if patient summary is too short."""
    if not isinstance(output_data, str):
        return {
            'check_id': 'L1_TOO_SHORT',
            'check_level': 1,
            'has_error': False,
            'error_reason': ''
        }

    word_count = len(output_data.split())
    has_error = word_count < 50
    return {
        'check_id': 'L1_TOO_SHORT',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': f'Summary too short: {word_count} words (minimum 50)' if has_error else ''
    }


def check_l1_too_long(output_data: str) -> dict:
    """L1: Check if patient summary is too long."""
    if not isinstance(output_data, str):
        return {
            'check_id': 'L1_TOO_LONG',
            'check_level': 1,
            'has_error': False,
            'error_reason': ''
        }

    word_count = len(output_data.split())
    has_error = word_count > 300
    return {
        'check_id': 'L1_TOO_LONG',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': f'Summary too long: {word_count} words (maximum 300)' if has_error else ''
    }


def check_l1_missing_sections(output_data: str) -> dict:
    """L1: Check for missing section headers in patient summary."""
    if not isinstance(output_data, str):
        return {
            'check_id': 'L1_MISSING_SECTIONS',
            'check_level': 1,
            'has_error': False,
            'error_reason': ''
        }

    # Expected emoji headers
    expected_sections = ['Clinical Status', 'Recent Activity', 'Tags', 'Upcoming']

    found = [s for s in expected_sections if s.lower() in output_data.lower()]

    has_error = len(found) == 0
    return {
        'check_id': 'L1_MISSING_SECTIONS',
        'check_level': 1,
        'has_error': has_error,
        'error_reason': 'No expected section headers found in summary' if has_error else ''
    }


def run_level1_checks(element_id: str, use_case: str, raw_data: dict, metadata: dict) -> list[dict]:
    """
    Run all Level 1 (rule-based) checks for a trace.

    Args:
        element_id: The trace element ID
        use_case: The use case name (e.g., 'worklist_rad_report_data_extraction')
        raw_data: The raw trace data
        metadata: Patient/workspace metadata

    Returns:
        List of check result dicts
    """
    results = []

    output_type, output_data = extract_output_from_trace(raw_data)
    messages = raw_data.get('request', {}).get('messages', [])

    # Generic checks (all use cases)
    results.append(check_l1_empty_response(output_type, output_data))
    results.append(check_l1_json_parse_error(output_type, output_data))
    results.append(check_l1_refusal(output_type, output_data))
    results.append(check_l1_patient_mismatch(output_data, metadata))

    # Use case specific checks
    if use_case == 'worklist_rad_report_data_extraction' and output_type == 'tool_call':
        results.append(check_l1_missing_heading(output_data))
        results.append(check_l1_missing_highlight(output_data))
        results.append(check_l1_unbalanced_highlight_tags(output_data))

        available_tags = parse_available_tags(messages)
        results.append(check_l1_invalid_tag_id(output_data, available_tags))

    elif use_case == 'worklist_suggest_next_steps' and output_type == 'tool_call':
        results.append(check_l1_invalid_action_type(output_data))
        results.append(check_l1_invalid_date(output_data))
        results.append(check_l1_missing_action_field(output_data))

    elif use_case == 'worklist_patient_summary' and output_type == 'content':
        results.append(check_l1_too_short(output_data))
        results.append(check_l1_too_long(output_data))
        results.append(check_l1_missing_sections(output_data))

    # Add element_id to all results
    for r in results:
        r['element_id'] = element_id

    return results


# Level 2 Checks (LLM-based)

def load_error_detection_prompts() -> dict:
    """Load error detection prompts from YAML file."""
    if not ERROR_DETECTION_PROMPTS.exists():
        logger.warning(f"Error detection prompts not found: {ERROR_DETECTION_PROMPTS}")
        return {}

    with open(ERROR_DETECTION_PROMPTS, 'r') as f:
        return yaml.safe_load(f)


def run_level2_checks(element_id: str, use_case: str, raw_data: dict, metadata: dict) -> list[dict]:
    """
    Run Level 2 (LLM-based) checks for a trace.

    Args:
        element_id: The trace element ID
        use_case: The use case name
        raw_data: The raw trace data
        metadata: Patient/workspace metadata

    Returns:
        List of check result dicts
    """
    results = []

    try:
        from llm_client import get_portkey_client, get_json_response
    except ImportError as e:
        logger.warning(f"Could not import llm_client: {e}")
        return results

    output_type, output_data = extract_output_from_trace(raw_data)
    input_text = extract_input_from_trace(raw_data)

    prompts_config = load_error_detection_prompts()
    if not prompts_config:
        return results

    prompts = {p['use_case']: p for p in prompts_config.get('prompts', [])}

    try:
        client = get_portkey_client()
    except Exception as e:
        logger.warning(f"Could not create Portkey client: {e}")
        return results

    # Prepare output string
    if output_type == 'tool_call':
        output_str = json.dumps(output_data, indent=2)
    elif output_type == 'content':
        output_str = output_data
    else:
        output_str = str(output_data)

    # L2_HALLUCINATION - Check for all use cases
    hallucination_prompt = prompts.get('error_detection_hallucination')
    if hallucination_prompt:
        try:
            system_prompt = hallucination_prompt['system_prompt']
            user_prompt = hallucination_prompt['user_prompt'].replace('{{INPUT}}', input_text).replace('{{OUTPUT}}', output_str)
            model_config = hallucination_prompt.get('model_config', {})

            response = get_json_response(
                client,
                system_prompt,
                user_prompt,
                model_config.get('model', 'openai/gpt-4o-mini'),
                {'temperature': model_config.get('temperature', 0), 'max_tokens': model_config.get('max_tokens', 100)}
            )

            if response:
                results.append({
                    'element_id': element_id,
                    'check_id': 'L2_HALLUCINATION',
                    'check_level': 2,
                    'has_error': response.get('has_error', False),
                    'error_reason': response.get('reason', '') if response.get('has_error') else ''
                })
        except Exception as e:
            logger.warning(f"L2_HALLUCINATION check failed for {element_id}: {e}")

    # L2_FORMAT_ERROR - Only for structured output use cases
    if use_case in ['worklist_rad_report_data_extraction', 'worklist_suggest_next_steps']:
        format_prompt = prompts.get('error_detection_format_compliance')
        output_schemas = prompts_config.get('output_schemas', {})
        expected_schema = output_schemas.get(use_case, {}).get('schema', '')

        if format_prompt and expected_schema:
            try:
                system_prompt = format_prompt['system_prompt']
                user_prompt = format_prompt['user_prompt'].replace('{{EXPECTED_FORMAT}}', expected_schema).replace('{{OUTPUT}}', output_str)
                model_config = format_prompt.get('model_config', {})

                response = get_json_response(
                    client,
                    system_prompt,
                    user_prompt,
                    model_config.get('model', 'openai/gpt-4o-mini'),
                    {'temperature': model_config.get('temperature', 0), 'max_tokens': model_config.get('max_tokens', 100)}
                )

                if response:
                    results.append({
                        'element_id': element_id,
                        'check_id': 'L2_FORMAT_ERROR',
                        'check_level': 2,
                        'has_error': response.get('has_error', False),
                        'error_reason': response.get('reason', '') if response.get('has_error') else ''
                    })
            except Exception as e:
                logger.warning(f"L2_FORMAT_ERROR check failed for {element_id}: {e}")

    # L2_CLINICAL_SAFETY - Only for suggest_next_steps
    if use_case == 'worklist_suggest_next_steps':
        safety_prompt = prompts.get('error_detection_clinical_safety')
        if safety_prompt:
            try:
                # Build patient context from metadata
                patient_context = f"Patient: {metadata.get('patient_name', 'Unknown')}\n"
                patient_context += f"Patient ID: {metadata.get('patient_id', 'Unknown')}\n"
                patient_context += f"Workspace: {metadata.get('workspace_name', 'Unknown')}"

                system_prompt = safety_prompt['system_prompt']
                user_prompt = safety_prompt['user_prompt'].replace('{{PATIENT_CONTEXT}}', patient_context).replace('{{OUTPUT}}', output_str)
                model_config = safety_prompt.get('model_config', {})

                response = get_json_response(
                    client,
                    system_prompt,
                    user_prompt,
                    model_config.get('model', 'openai/gpt-4o-mini'),
                    {'temperature': model_config.get('temperature', 0), 'max_tokens': model_config.get('max_tokens', 100)}
                )

                if response:
                    results.append({
                        'element_id': element_id,
                        'check_id': 'L2_CLINICAL_SAFETY',
                        'check_level': 2,
                        'has_error': response.get('has_error', False),
                        'error_reason': response.get('reason', '') if response.get('has_error') else ''
                    })
            except Exception as e:
                logger.warning(f"L2_CLINICAL_SAFETY check failed for {element_id}: {e}")

    return results


def store_check_results(results: list[dict], conn: sqlite3.Connection = None):
    """Store check results in the trace_errors table."""
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    cursor = conn.cursor()

    for result in results:
        cursor.execute('''
            INSERT OR REPLACE INTO trace_errors (element_id, check_id, check_level, has_error, error_reason, created_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        ''', (
            result['element_id'],
            result['check_id'],
            result['check_level'],
            1 if result['has_error'] else 0,
            result.get('error_reason', '')
        ))

    conn.commit()

    if should_close:
        conn.close()


def run_all_checks(element_id: str, run_level2: bool = False) -> list[dict]:
    """
    Run all error checks for a single trace element.

    Args:
        element_id: The trace element ID
        run_level2: Whether to run LLM-based Level 2 checks

    Returns:
        List of all check results
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get trace data
    cursor.execute('''
        SELECT t.system_prompt_hash, r.data
        FROM traces t
        JOIN raw_traces r ON t.element_id = r.element_id
        WHERE t.element_id = ?
    ''', (element_id,))

    row = cursor.fetchone()
    if not row:
        conn.close()
        return []

    prompt_hash, raw_data_str = row
    raw_data = json.loads(raw_data_str)

    # Determine use case from prompt hash
    use_case = HASH_TO_USE_CASE.get(prompt_hash)
    if not use_case:
        conn.close()
        return []

    # Get metadata
    metadata = get_trace_metadata(element_id) or {}

    # Run Level 1 checks
    results = run_level1_checks(element_id, use_case, raw_data, metadata)

    # Run Level 2 checks if requested
    if run_level2:
        l2_results = run_level2_checks(element_id, use_case, raw_data, metadata)
        results.extend(l2_results)

    # Store results
    init_error_detection_table(conn)
    store_check_results(results, conn)

    conn.close()
    return results


def process_single_trace(args: tuple) -> tuple:
    """Process a single trace for parallel execution."""
    element_id, prompt_hash, raw_data_str, run_level2 = args

    try:
        raw_data = json.loads(raw_data_str)
        use_case = HASH_TO_USE_CASE.get(prompt_hash)

        if not use_case:
            return element_id, [], None

        metadata = get_trace_metadata(element_id) or {}

        # Run checks
        results = run_level1_checks(element_id, use_case, raw_data, metadata)

        if run_level2:
            l2_results = run_level2_checks(element_id, use_case, raw_data, metadata)
            results.extend(l2_results)

        return element_id, results, use_case

    except Exception as e:
        logger.error(f"Error processing trace {element_id}: {e}")
        return element_id, [], None


def batch_run_checks(use_case_filter: str = None, limit: int = None, run_level2: bool = False, max_workers: int = 10) -> dict:
    """
    Run error checks on multiple traces in batch with parallel processing.

    Args:
        use_case_filter: Optional use case name to filter by
        limit: Maximum number of traces to process
        run_level2: Whether to run LLM-based Level 2 checks
        max_workers: Number of parallel workers (default 10)

    Returns:
        Stats dict with processing results
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    init_error_detection_table(conn)

    # Build query to get traces with metadata
    sql = '''
        SELECT t.element_id, t.system_prompt_hash, r.data
        FROM traces t
        JOIN raw_traces r ON t.element_id = r.element_id
        JOIN trace_metadata m ON t.element_id = m.element_id
    '''

    params = []
    conditions = []

    # Filter by use case if specified
    if use_case_filter:
        # Get prompt hashes for this use case
        hashes = [h for h, uc in HASH_TO_USE_CASE.items() if uc == use_case_filter]
        if hashes:
            placeholders = ','.join(['?' for _ in hashes])
            conditions.append(f't.system_prompt_hash IN ({placeholders})')
            params.extend(hashes)
        else:
            return {'error': f'Unknown use case: {use_case_filter}'}

    if conditions:
        sql += ' WHERE ' + ' AND '.join(conditions)

    sql += ' ORDER BY t.parsed_timestamp DESC'

    if limit:
        sql += ' LIMIT ?'
        params.append(limit)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    logger.info(f"Processing {len(rows)} traces for error detection with {max_workers} workers")

    stats = {
        'total_processed': 0,
        'total_errors': 0,
        'l1_errors': 0,
        'l2_errors': 0,
        'by_check': {},
        'by_use_case': {}
    }

    # Prepare args for parallel processing
    task_args = [(row[0], row[1], row[2], run_level2) for row in rows]

    # Process in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_trace, args): args[0] for args in task_args}

        completed = 0
        for future in as_completed(futures):
            element_id, results, use_case = future.result()
            completed += 1

            if results:
                all_results.append((element_id, results, use_case))

                # Update stats
                stats['total_processed'] += 1

                for r in results:
                    check_id = r['check_id']
                    if check_id not in stats['by_check']:
                        stats['by_check'][check_id] = {'total': 0, 'errors': 0}
                    stats['by_check'][check_id]['total'] += 1

                    if r['has_error']:
                        stats['total_errors'] += 1
                        stats['by_check'][check_id]['errors'] += 1

                        if r['check_level'] == 1:
                            stats['l1_errors'] += 1
                        else:
                            stats['l2_errors'] += 1

                # Track by use case
                if use_case:
                    if use_case not in stats['by_use_case']:
                        stats['by_use_case'][use_case] = {'total': 0, 'errors': 0}
                    stats['by_use_case'][use_case]['total'] += 1
                    if any(r['has_error'] for r in results):
                        stats['by_use_case'][use_case]['errors'] += 1

            if completed % 100 == 0:
                logger.info(f"Processed {completed}/{len(rows)} traces")

    # Store all results in batch
    logger.info(f"Storing {len(all_results)} results to database...")
    conn = get_db_connection()
    for element_id, results, use_case in all_results:
        store_check_results(results, conn)
    conn.commit()
    conn.close()

    logger.info(f"Batch processing complete: {stats['total_processed']} traces, {stats['total_errors']} errors found")
    return stats


def get_errors_for_element(element_id: str) -> list[dict]:
    """Get all error check results for a specific trace element."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT check_id, check_level, has_error, error_reason, created_at
        FROM trace_errors
        WHERE element_id = ?
        ORDER BY check_level, check_id
    ''', (element_id,))

    results = []
    for row in cursor.fetchall():
        results.append({
            'check_id': row[0],
            'check_level': row[1],
            'has_error': bool(row[2]),
            'error_reason': row[3],
            'created_at': row[4]
        })

    conn.close()
    return results


def get_error_stats() -> dict:
    """Get aggregated error detection statistics for dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Total traces checked
        cursor.execute('SELECT COUNT(DISTINCT element_id) FROM trace_errors')
        total_checked = cursor.fetchone()[0]

        # Traces with errors
        cursor.execute('SELECT COUNT(DISTINCT element_id) FROM trace_errors WHERE has_error = 1')
        traces_with_errors = cursor.fetchone()[0]

        # Errors by check_id
        cursor.execute('''
            SELECT check_id, check_level, COUNT(*) as total, SUM(has_error) as errors
            FROM trace_errors
            GROUP BY check_id, check_level
            ORDER BY errors DESC
        ''')
        by_check = []
        for row in cursor.fetchall():
            by_check.append({
                'check_id': row[0],
                'check_level': row[1],
                'total': row[2],
                'errors': row[3],
                'error_rate': round((row[3] / row[2]) * 100, 1) if row[2] > 0 else 0
            })

        # Errors by level
        cursor.execute('''
            SELECT check_level, COUNT(*) as total, SUM(has_error) as errors
            FROM trace_errors
            GROUP BY check_level
        ''')
        by_level = {}
        for row in cursor.fetchall():
            by_level[f'L{row[0]}'] = {'total': row[1], 'errors': row[2]}

        conn.close()

        return {
            'total_checked': total_checked,
            'traces_with_errors': traces_with_errors,
            'error_rate': round((traces_with_errors / total_checked) * 100, 1) if total_checked > 0 else 0,
            'by_check': by_check,
            'by_level': by_level
        }

    except sqlite3.OperationalError:
        conn.close()
        return {
            'total_checked': 0,
            'traces_with_errors': 0,
            'error_rate': 0,
            'by_check': [],
            'by_level': {}
        }


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Run error detection on traces')
    parser.add_argument('--use-case', type=str, help='Filter by use case')
    parser.add_argument('--limit', type=int, help='Maximum traces to process')
    parser.add_argument('--level2', action='store_true', help='Run Level 2 LLM checks')
    parser.add_argument('--element', type=str, help='Run checks on specific element_id')
    parser.add_argument('--stats', action='store_true', help='Show error detection stats')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers (default: 10)')
    args = parser.parse_args()

    if args.stats:
        stats = get_error_stats()
        print("\nError Detection Statistics:")
        print(f"  Total traces checked: {stats['total_checked']}")
        print(f"  Traces with errors: {stats['traces_with_errors']} ({stats['error_rate']}%)")
        print("\nBy check:")
        for check in stats['by_check']:
            print(f"  {check['check_id']}: {check['errors']}/{check['total']} ({check['error_rate']}%)")
    elif args.element:
        results = run_all_checks(args.element, run_level2=args.level2)
        print(f"\nResults for {args.element}:")
        for r in results:
            status = "ERROR" if r['has_error'] else "OK"
            print(f"  [{r['check_level']}] {r['check_id']}: {status}")
            if r['has_error'] and r.get('error_reason'):
                print(f"      -> {r['error_reason']}")
    else:
        stats = batch_run_checks(
            use_case_filter=args.use_case,
            limit=args.limit,
            run_level2=args.level2,
            max_workers=args.workers
        )
        print(f"\nBatch processing complete:")
        print(f"  Processed: {stats['total_processed']}")
        print(f"  Errors found: {stats['total_errors']}")
        print(f"  L1 errors: {stats['l1_errors']}")
        print(f"  L2 errors: {stats['l2_errors']}")
