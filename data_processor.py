import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional
import sqlite3

from config import JSONL_FILE, PROMPTS_DIR, PROCESSED_DIR, TRACES_DB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TraceInfo:
    trace_id: str
    element_id: str  # Unique ID for this specific element (from 'id' field)
    created_at: str
    parsed_timestamp: Optional[str]
    model: str
    cost: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    response_time: int
    system_prompt_hash: str
    system_prompt_file: str
    num_messages: int
    num_turns: int
    has_tool_calls: bool
    has_metadata: bool
    metadata_keys: str
    is_multi_turn: bool
    user_disagreement_detected: bool
    potential_error_indicators: str


def parse_timestamp(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    formats = [
        '%a %b %d %Y %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
    ]
    # Strip timezone info for parsing
    clean_date = date_str.split(' GMT')[0] if ' GMT' in date_str else date_str
    for fmt in formats:
        try:
            return datetime.strptime(clean_date, fmt)
        except ValueError:
            continue
    return None


def extract_system_prompt(messages: list) -> Optional[str]:
    for msg in messages:
        if msg.get('role') == 'system':
            return msg.get('content', '')
    return None


def count_turns(messages: list) -> int:
    return sum(1 for msg in messages if msg.get('role') == 'user')


def has_tool_calls(messages: list, response: dict) -> bool:
    for msg in messages:
        if msg.get('tool_calls'):
            return True
    choices = response.get('choices', [])
    if choices:
        msg = choices[0].get('message', {})
        if msg.get('tool_calls'):
            return True
    return False


def detect_user_disagreement(messages: list) -> bool:
    """Detect if user disagreed with LLM output in multi-turn conversation."""
    disagreement_phrases = [
        'no, ', 'no.', 'that\'s wrong', 'incorrect', 'not right',
        'actually,', 'that\'s not what i', 'you misunderstood',
        'please try again', 'that\'s not correct', 'wrong answer',
        'not what i asked', 'i said', 'i meant', 'let me clarify'
    ]
    for i, msg in enumerate(messages):
        if msg.get('role') == 'user' and i > 0:
            content = msg.get('content', '').lower()
            for phrase in disagreement_phrases:
                if phrase in content:
                    return True
    return False


def detect_error_indicators(trace: dict) -> list[str]:
    """Detect potential error indicators in a trace."""
    indicators = []

    # Check response status
    if trace.get('response_status_code', 200) != 200:
        indicators.append(f"status_code_{trace.get('response_status_code')}")

    # Check for error in response
    response = trace.get('response', {})
    if 'error' in response:
        indicators.append('response_error')

    # Check for empty response
    choices = response.get('choices', [])
    if not choices:
        indicators.append('empty_choices')
    elif choices:
        content = choices[0].get('message', {}).get('content', '')
        if not content and not choices[0].get('message', {}).get('tool_calls'):
            indicators.append('empty_content')

    # Check for refusal patterns
    if choices:
        content = choices[0].get('message', {}).get('content', '').lower()
        refusal_patterns = ["i can't", "i cannot", "i'm unable", "i apologize"]
        for pattern in refusal_patterns:
            if pattern in content:
                indicators.append('potential_refusal')
                break

    return indicators


def extract_metadata_info(trace: dict) -> tuple[bool, list[str]]:
    """Extract metadata presence and keys from trace."""
    metadata_keys = []

    # Check metadata field
    metadata = trace.get('metadata', {})
    if metadata and isinstance(metadata, dict):
        metadata_keys.extend([f"metadata.{k}" for k in metadata.keys()])

    # Check portkeyHeaders
    portkey_headers = trace.get('portkeyHeaders', {})
    if portkey_headers and isinstance(portkey_headers, dict):
        for key, value in portkey_headers.items():
            if 'metadata' in key.lower():
                metadata_keys.append(f"portkeyHeaders.{key}")
                if isinstance(value, dict) and value:
                    metadata_keys.extend([f"portkeyHeaders.{key}.{k}" for k in value.keys()])

    # Check x-portkey-metadata specifically
    if 'x-portkey-metadata' in trace.get('portkeyHeaders', {}):
        val = trace['portkeyHeaders']['x-portkey-metadata']
        if val and isinstance(val, dict):
            metadata_keys.extend([f"x-portkey-metadata.{k}" for k in val.keys()])

    has_meaningful_metadata = len([k for k in metadata_keys if not k.endswith('x-portkey-config')]) > 0
    return has_meaningful_metadata, metadata_keys


def save_system_prompt(content: str, prompt_hash: str) -> str:
    """Save system prompt to file and return filename."""
    filename = f"prompt_{prompt_hash}.txt"
    filepath = PROMPTS_DIR / filename

    if not filepath.exists():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved new prompt: {filename}")

    return filename


def process_trace(trace: dict) -> Optional[TraceInfo]:
    """Process a single trace and extract all relevant information."""
    try:
        request = trace.get('request', {})
        response = trace.get('response', {})
        messages = request.get('messages', [])

        # Extract system prompt
        system_prompt = extract_system_prompt(messages)
        if system_prompt:
            prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]
            prompt_file = save_system_prompt(system_prompt, prompt_hash)
        else:
            prompt_hash = "no_system_prompt"
            prompt_file = ""

        # Extract usage info
        usage = response.get('usage', {})

        # Count messages and turns
        num_messages = len(messages)
        num_turns = count_turns(messages)
        is_multi_turn = num_turns > 1

        # Parse timestamp
        created_at = trace.get('created_at', '')
        parsed_ts = parse_timestamp(created_at)

        # Detect metadata
        has_meta, meta_keys = extract_metadata_info(trace)

        # Detect errors and user disagreement
        user_disagreed = detect_user_disagreement(messages) if is_multi_turn else False
        error_indicators = detect_error_indicators(trace)

        return TraceInfo(
            trace_id=trace.get('trace_id', ''),
            element_id=trace.get('id', ''),  # Unique element ID
            created_at=created_at,
            parsed_timestamp=parsed_ts.isoformat() if parsed_ts else None,
            model=trace.get('ai_model', request.get('model', 'unknown')),
            cost=trace.get('cost', 0.0),
            total_tokens=usage.get('total_tokens', 0),
            prompt_tokens=usage.get('prompt_tokens', 0),
            completion_tokens=usage.get('completion_tokens', 0),
            response_time=trace.get('response_time', 0),
            system_prompt_hash=prompt_hash,
            system_prompt_file=prompt_file,
            num_messages=num_messages,
            num_turns=num_turns,
            has_tool_calls=has_tool_calls(messages, response),
            has_metadata=has_meta,
            metadata_keys=','.join(meta_keys),
            is_multi_turn=is_multi_turn,
            user_disagreement_detected=user_disagreed,
            potential_error_indicators=','.join(error_indicators)
        )
    except Exception as e:
        logger.error(f"Error processing trace: {e}")
        return None


def init_database():
    """Initialize SQLite database for storing processed traces."""
    conn = sqlite3.connect(TRACES_DB)
    cursor = conn.cursor()

    # Drop old tables to recreate with new schema
    cursor.execute('DROP TABLE IF EXISTS traces')
    cursor.execute('DROP TABLE IF EXISTS raw_traces')
    cursor.execute('DROP TABLE IF EXISTS prompts')

    cursor.execute('''
        CREATE TABLE traces (
            element_id TEXT PRIMARY KEY,
            trace_id TEXT,
            created_at TEXT,
            parsed_timestamp TEXT,
            model TEXT,
            cost REAL,
            total_tokens INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            response_time INTEGER,
            system_prompt_hash TEXT,
            system_prompt_file TEXT,
            num_messages INTEGER,
            num_turns INTEGER,
            has_tool_calls INTEGER,
            has_metadata INTEGER,
            metadata_keys TEXT,
            is_multi_turn INTEGER,
            user_disagreement_detected INTEGER,
            potential_error_indicators TEXT
        )
    ''')

    # Index on trace_id for grouping
    cursor.execute('CREATE INDEX idx_trace_id ON traces(trace_id)')
    cursor.execute('CREATE INDEX idx_system_prompt_hash ON traces(system_prompt_hash)')
    cursor.execute('CREATE INDEX idx_model ON traces(model)')
    cursor.execute('CREATE INDEX idx_parsed_timestamp ON traces(parsed_timestamp)')

    cursor.execute('''
        CREATE TABLE prompts (
            prompt_hash TEXT PRIMARY KEY,
            filename TEXT,
            trace_count INTEGER,
            first_seen TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE raw_traces (
            element_id TEXT PRIMARY KEY,
            trace_id TEXT,
            data TEXT
        )
    ''')
    cursor.execute('CREATE INDEX idx_raw_trace_id ON raw_traces(trace_id)')

    conn.commit()
    return conn


def process_jsonl(sample_size: int = 1000) -> dict:
    """Process JSONL file and store results in database."""
    logger.info(f"Processing {sample_size} traces from {JSONL_FILE}")

    conn = init_database()
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute('DELETE FROM traces')
    cursor.execute('DELETE FROM prompts')
    cursor.execute('DELETE FROM raw_traces')
    conn.commit()

    processed_count = 0
    error_count = 0
    prompt_counts = Counter()

    with open(JSONL_FILE, 'r') as f:
        for line in f:
            if processed_count >= sample_size:
                break

            try:
                trace = json.loads(line)
                trace_info = process_trace(trace)

                if trace_info:
                    # Insert trace info (element_id as primary key)
                    cursor.execute('''
                        INSERT OR REPLACE INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trace_info.element_id, trace_info.trace_id,
                        trace_info.created_at, trace_info.parsed_timestamp,
                        trace_info.model, trace_info.cost, trace_info.total_tokens,
                        trace_info.prompt_tokens, trace_info.completion_tokens, trace_info.response_time,
                        trace_info.system_prompt_hash, trace_info.system_prompt_file,
                        trace_info.num_messages, trace_info.num_turns,
                        1 if trace_info.has_tool_calls else 0,
                        1 if trace_info.has_metadata else 0,
                        trace_info.metadata_keys,
                        1 if trace_info.is_multi_turn else 0,
                        1 if trace_info.user_disagreement_detected else 0,
                        trace_info.potential_error_indicators
                    ))

                    # Store raw trace for viewing (element_id as primary key)
                    cursor.execute('''
                        INSERT OR REPLACE INTO raw_traces VALUES (?, ?, ?)
                    ''', (trace_info.element_id, trace_info.trace_id, json.dumps(trace)))

                    prompt_counts[trace_info.system_prompt_hash] += 1
                    processed_count += 1
                else:
                    error_count += 1

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                error_count += 1

            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} traces...")
                conn.commit()

    # Update prompt counts
    for prompt_hash, count in prompt_counts.items():
        cursor.execute('''
            INSERT OR REPLACE INTO prompts (prompt_hash, filename, trace_count, first_seen)
            VALUES (?, ?, ?, datetime('now'))
        ''', (prompt_hash, f"prompt_{prompt_hash}.txt", count))

    conn.commit()
    conn.close()

    logger.info(f"Completed processing: {processed_count} traces, {error_count} errors")
    logger.info(f"Unique prompts: {len(prompt_counts)}")

    return {
        'processed': processed_count,
        'errors': error_count,
        'unique_prompts': len(prompt_counts)
    }


if __name__ == "__main__":
    from config import TEST_SAMPLE_SIZE
    result = process_jsonl(TEST_SAMPLE_SIZE)
    print(f"\nProcessing complete: {result}")
