import json
import hashlib
import logging
import time
import os
from pathlib import Path
from datetime import datetime
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional
import sqlite3

from config import JSONL_FILE, PROMPTS_DIR, PROCESSED_DIR, TRACES_DB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


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
    logger.info(f"Initializing database at {TRACES_DB}")

    # Check if database exists and get its size
    if os.path.exists(TRACES_DB):
        old_size = os.path.getsize(TRACES_DB)
        logger.info(f"Existing database found ({format_file_size(old_size)}), will be recreated")

    conn = sqlite3.connect(TRACES_DB)
    cursor = conn.cursor()

    # Drop old tables to recreate with new schema
    logger.info("Dropping existing tables...")
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
    logger.info("Database schema created successfully")
    return conn


def is_progressive_build(elements: list) -> bool:
    """
    Detect if a group of elements represents a progressive conversation build.

    Progressive build pattern:
    - Message counts are non-decreasing when sorted by time (allows duplicates)
    - Last element has more messages than first (conversation grew)
    - Same conversation being built up (later elements contain earlier messages)

    Separate elements pattern:
    - Message counts decrease at some point (e.g., 5, 5, 5 - retries with same content)
    - Or all elements have same message count
    - Different interactions with same trace_id
    """
    if len(elements) <= 1:
        return False

    # Sort by parsed timestamp
    sorted_elements = sorted(elements, key=lambda x: x['parsed_timestamp'] or '')

    # Get message counts in time order
    msg_counts = [e['num_messages'] for e in sorted_elements]

    # Check if non-decreasing (allows duplicates but no decreases)
    is_non_decreasing = all(msg_counts[i] <= msg_counts[i+1] for i in range(len(msg_counts)-1))

    # Also require that the conversation actually grew (last > first)
    conversation_grew = msg_counts[-1] > msg_counts[0]

    return is_non_decreasing and conversation_grew


def filter_progressive_builds(trace_groups: dict) -> list:
    """
    Filter trace groups to handle progressive builds.

    For progressive builds: keep only the latest element (has all messages)
    For separate elements: keep all elements

    Returns list of (trace_info, raw_trace) tuples to insert.
    """
    filtered = []
    progressive_count = 0
    elements_removed = 0

    for trace_id, elements in trace_groups.items():
        if len(elements) == 1:
            # Single element, keep it
            filtered.append(elements[0])
        elif is_progressive_build(elements):
            # Progressive build - keep only the latest (most messages)
            progressive_count += 1
            elements_removed += len(elements) - 1
            # Sort by timestamp and keep the last one
            sorted_elements = sorted(elements, key=lambda x: x['parsed_timestamp'] or '')
            filtered.append(sorted_elements[-1])
        else:
            # Separate elements - keep all
            filtered.extend(elements)

    if progressive_count > 0:
        logger.info(f"Detected {progressive_count} progressive conversation builds")
        logger.info(f"Removed {elements_removed} intermediate elements (keeping latest with full conversation)")

    return filtered


def process_jsonl(sample_size: int = 1000) -> dict:
    """Process JSONL file and store results in database."""
    start_time = time.time()

    # Log file info
    if os.path.exists(JSONL_FILE):
        file_size = os.path.getsize(JSONL_FILE)
        logger.info(f"=" * 60)
        logger.info(f"STARTING TRACE PROCESSING")
        logger.info(f"=" * 60)
        logger.info(f"Input file: {JSONL_FILE}")
        logger.info(f"File size: {format_file_size(file_size)}")
        logger.info(f"Sample size: {sample_size:,} traces")
        logger.info(f"-" * 60)
    else:
        logger.error(f"Input file not found: {JSONL_FILE}")
        return {'processed': 0, 'errors': 0, 'unique_prompts': 0, 'error': 'File not found'}

    # Phase 1: Read all traces and group by trace_id
    logger.info("Phase 1: Reading and grouping traces by trace_id...")
    trace_groups = {}  # trace_id -> list of (trace_info, raw_trace)
    read_count = 0
    error_count = 0
    batch_start_time = time.time()

    with open(JSONL_FILE, 'r') as f:
        for line in f:
            if read_count >= sample_size:
                break

            try:
                trace = json.loads(line)
                trace_info = process_trace(trace)

                if trace_info:
                    # Group by trace_id
                    if trace_info.trace_id not in trace_groups:
                        trace_groups[trace_info.trace_id] = []
                    trace_groups[trace_info.trace_id].append({
                        'trace_info': trace_info,
                        'raw_trace': trace,
                        'parsed_timestamp': trace_info.parsed_timestamp,
                        'num_messages': trace_info.num_messages
                    })
                    read_count += 1
                else:
                    error_count += 1

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error at line {read_count + error_count + 1}: {e}")
                error_count += 1

            if read_count % 1000 == 0 and read_count > 0:
                batch_elapsed = time.time() - batch_start_time
                rate = 1000 / batch_elapsed if batch_elapsed > 0 else 0
                progress_pct = (read_count / sample_size) * 100
                logger.info(f"Reading: {read_count:,}/{sample_size:,} ({progress_pct:.1f}%) - {rate:.0f} traces/sec")
                batch_start_time = time.time()

    logger.info(f"Read {read_count:,} elements across {len(trace_groups):,} unique trace_ids")

    # Phase 2: Filter progressive builds
    logger.info("-" * 60)
    logger.info("Phase 2: Detecting and filtering progressive conversation builds...")
    filtered_elements = filter_progressive_builds(trace_groups)
    logger.info(f"After filtering: {len(filtered_elements):,} elements to insert")

    # Phase 3: Insert filtered traces into database
    logger.info("-" * 60)
    logger.info("Phase 3: Inserting filtered traces into database...")

    conn = init_database()
    cursor = conn.cursor()

    # Clear existing data
    logger.info("Clearing existing data...")
    cursor.execute('DELETE FROM traces')
    cursor.execute('DELETE FROM prompts')
    cursor.execute('DELETE FROM raw_traces')
    conn.commit()
    logger.info("Existing data cleared")

    processed_count = 0
    prompt_counts = Counter()
    model_counts = Counter()
    batch_start_time = time.time()

    for element in filtered_elements:
        trace_info = element['trace_info']
        raw_trace = element['raw_trace']

        # Insert trace info
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

        # Store raw trace
        cursor.execute('''
            INSERT OR REPLACE INTO raw_traces VALUES (?, ?, ?)
        ''', (trace_info.element_id, trace_info.trace_id, json.dumps(raw_trace)))

        prompt_counts[trace_info.system_prompt_hash] += 1
        model_counts[trace_info.model] += 1
        processed_count += 1

        if processed_count % 1000 == 0:
            batch_elapsed = time.time() - batch_start_time
            rate = 1000 / batch_elapsed if batch_elapsed > 0 else 0
            progress_pct = (processed_count / len(filtered_elements)) * 100
            logger.info(f"Inserting: {processed_count:,}/{len(filtered_elements):,} ({progress_pct:.1f}%) - {rate:.0f} traces/sec")
            conn.commit()
            batch_start_time = time.time()

    # Update prompt counts
    logger.info("Saving prompt statistics...")
    for prompt_hash, count in prompt_counts.items():
        cursor.execute('''
            INSERT OR REPLACE INTO prompts (prompt_hash, filename, trace_count, first_seen)
            VALUES (?, ?, ?, datetime('now'))
        ''', (prompt_hash, f"prompt_{prompt_hash}.txt", count))

    conn.commit()

    # Get final database size
    db_size = os.path.getsize(TRACES_DB) if os.path.exists(TRACES_DB) else 0

    conn.close()

    # Calculate final stats
    elapsed_time = time.time() - start_time
    avg_rate = processed_count / elapsed_time if elapsed_time > 0 else 0
    elements_filtered = read_count - processed_count

    # Log summary
    logger.info(f"-" * 60)
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"-" * 60)
    logger.info(f"Total elements read: {read_count:,}")
    logger.info(f"Progressive builds filtered: {elements_filtered:,}")
    logger.info(f"Final elements in DB: {processed_count:,}")
    logger.info(f"Unique trace IDs: {len(trace_groups):,}")
    logger.info(f"Processing errors: {error_count:,}")
    logger.info(f"Unique system prompts: {len(prompt_counts)}")
    logger.info(f"Unique models: {len(model_counts)}")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Average rate: {avg_rate:.0f} traces/second")
    logger.info(f"Database size: {format_file_size(db_size)}")

    # Log model distribution
    logger.info(f"-" * 60)
    logger.info("MODEL DISTRIBUTION:")
    for model, count in model_counts.most_common():
        pct = (count / processed_count) * 100 if processed_count > 0 else 0
        logger.info(f"  {model}: {count:,} ({pct:.1f}%)")

    # Log prompt distribution
    logger.info(f"-" * 60)
    logger.info("USE CASE DISTRIBUTION:")
    for prompt_hash, count in prompt_counts.most_common(5):
        pct = (count / processed_count) * 100 if processed_count > 0 else 0
        logger.info(f"  {prompt_hash[:16]}...: {count:,} ({pct:.1f}%)")

    logger.info(f"=" * 60)

    return {
        'processed': processed_count,
        'read': read_count,
        'filtered': elements_filtered,
        'unique_traces': len(trace_groups),
        'errors': error_count,
        'unique_prompts': len(prompt_counts),
        'elapsed_time': round(elapsed_time, 2),
        'traces_per_second': round(avg_rate, 0)
    }


if __name__ == "__main__":
    import argparse
    from config import SAMPLE_SIZE

    parser = argparse.ArgumentParser(description="Process LLM traces from JSONL file")
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE, help=f"Sample size (default: {SAMPLE_SIZE} from config.py)")
    args = parser.parse_args()

    result = process_jsonl(args.size)
    print(f"\nProcessing complete: {result}")
