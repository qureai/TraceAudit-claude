"""
Metadata Loader Module

Manages the trace_metadata table that stores patient/workspace information
from matched JSON files. This table is independent of the traces table
and can be reloaded at any time without reprocessing traces.

Usage:
    from metadata_loader import load_all_metadata, get_trace_metadata, get_available_workspaces

    # Load metadata from all matched_*.json files in data/
    load_all_metadata()

    # Get metadata for a specific element
    metadata = get_trace_metadata(element_id)

    # Get list of available workspaces for filtering
    workspaces = get_available_workspaces()
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from config import TRACES_DB, DATA_DIR

logger = logging.getLogger(__name__)

# Pattern for matched metadata files
MATCHED_METADATA_PATTERN = "matched_metadata_*.json"


def get_db_connection():
    """Get SQLite connection."""
    return sqlite3.connect(TRACES_DB)


def init_metadata_table(conn: sqlite3.Connection = None):
    """
    Initialize the trace_metadata table.
    This table stores patient/workspace info keyed by element_id.
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    cursor = conn.cursor()

    # Drop and recreate to ensure clean state
    cursor.execute('DROP TABLE IF EXISTS trace_metadata')

    cursor.execute('''
        CREATE TABLE trace_metadata (
            element_id TEXT PRIMARY KEY,
            trace_id TEXT,
            use_case TEXT,
            patient_id TEXT,
            patient_name TEXT,
            patient_pk INTEGER,
            workspace_id INTEGER,
            workspace_name TEXT,
            replica_source TEXT,
            match_success INTEGER,
            match_error TEXT
        )
    ''')

    # Create indexes for common query patterns
    cursor.execute('CREATE INDEX idx_metadata_trace_id ON trace_metadata(trace_id)')
    cursor.execute('CREATE INDEX idx_metadata_workspace_name ON trace_metadata(workspace_name)')
    cursor.execute('CREATE INDEX idx_metadata_patient_id ON trace_metadata(patient_id)')
    cursor.execute('CREATE INDEX idx_metadata_match_success ON trace_metadata(match_success)')

    conn.commit()
    logger.info("trace_metadata table initialized")

    if should_close:
        conn.close()


def load_metadata_file(filepath: Path, conn: sqlite3.Connection) -> tuple[int, int]:
    """
    Load a single matched metadata JSON file into the trace_metadata table.

    Args:
        filepath: Path to the matched_metadata_*.json file
        conn: SQLite connection

    Returns:
        Tuple of (records_loaded, records_skipped)
    """
    cursor = conn.cursor()
    loaded = 0
    skipped = 0

    try:
        with open(filepath, 'r') as f:
            records = json.load(f)

        for record in records:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO trace_metadata
                    (element_id, trace_id, use_case, patient_id, patient_name,
                     patient_pk, workspace_id, workspace_name, replica_source,
                     match_success, match_error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.get('element_id'),
                    record.get('trace_id'),
                    record.get('use_case'),
                    record.get('patient_id'),
                    record.get('patient_name'),
                    record.get('patient_pk'),
                    record.get('workspace_id'),
                    record.get('workspace_name'),
                    record.get('replica_source'),
                    1 if record.get('match_success') else 0,
                    record.get('match_error')
                ))
                loaded += 1
            except Exception as e:
                logger.debug(f"Error inserting record: {e}")
                skipped += 1

        conn.commit()

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {filepath}: {e}")
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")

    return loaded, skipped


def load_all_metadata() -> dict:
    """
    Load all matched_metadata_*.json files from the data directory.

    Returns:
        dict with stats about loading
    """
    # Find all matched metadata files
    matched_files = list(DATA_DIR.glob(MATCHED_METADATA_PATTERN))

    if not matched_files:
        logger.warning(f"No matched metadata files found in {DATA_DIR}")
        return {
            'files_found': 0,
            'total_loaded': 0,
            'total_skipped': 0
        }

    logger.info(f"Found {len(matched_files)} matched metadata files")

    conn = get_db_connection()

    # Initialize table (recreate to ensure clean state)
    init_metadata_table(conn)

    total_loaded = 0
    total_skipped = 0
    file_stats = {}

    for filepath in matched_files:
        logger.info(f"Loading {filepath.name}...")
        loaded, skipped = load_metadata_file(filepath, conn)
        file_stats[filepath.name] = {'loaded': loaded, 'skipped': skipped}
        total_loaded += loaded
        total_skipped += skipped
        logger.info(f"  Loaded {loaded} records ({skipped} skipped)")

    conn.close()

    logger.info(f"Total: {total_loaded} records loaded across {len(matched_files)} files")

    return {
        'files_found': len(matched_files),
        'total_loaded': total_loaded,
        'total_skipped': total_skipped,
        'file_stats': file_stats
    }


def get_trace_metadata(element_id: str) -> Optional[dict]:
    """
    Get metadata for a specific trace element.

    Args:
        element_id: The element ID to look up

    Returns:
        dict with metadata fields, or None if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT element_id, trace_id, use_case, patient_id, patient_name,
                   patient_pk, workspace_id, workspace_name, replica_source,
                   match_success, match_error
            FROM trace_metadata
            WHERE element_id = ?
        ''', (element_id,))

        row = cursor.fetchone()
        if row:
            return {
                'element_id': row[0],
                'trace_id': row[1],
                'use_case': row[2],
                'patient_id': row[3],
                'patient_name': row[4],
                'patient_pk': row[5],
                'workspace_id': row[6],
                'workspace_name': row[7],
                'replica_source': row[8],
                'match_success': bool(row[9]),
                'match_error': row[10]
            }
        return None

    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return None
    finally:
        conn.close()


def get_metadata_for_trace_id(trace_id: str) -> list[dict]:
    """
    Get all metadata records for a given trace_id.

    Args:
        trace_id: The trace ID to look up

    Returns:
        List of metadata dicts for all elements with this trace_id
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT element_id, trace_id, use_case, patient_id, patient_name,
                   patient_pk, workspace_id, workspace_name, replica_source,
                   match_success, match_error
            FROM trace_metadata
            WHERE trace_id = ?
        ''', (trace_id,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'element_id': row[0],
                'trace_id': row[1],
                'use_case': row[2],
                'patient_id': row[3],
                'patient_name': row[4],
                'patient_pk': row[5],
                'workspace_id': row[6],
                'workspace_name': row[7],
                'replica_source': row[8],
                'match_success': bool(row[9]),
                'match_error': row[10]
            })
        return results

    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def get_available_workspaces() -> list[dict]:
    """
    Get list of unique workspaces with their trace counts.

    Returns:
        List of dicts with workspace_name and count
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT workspace_name, COUNT(*) as count
            FROM trace_metadata
            WHERE workspace_name IS NOT NULL AND workspace_name != ''
            GROUP BY workspace_name
            ORDER BY count DESC
        ''')

        return [{'workspace_name': row[0], 'count': row[1]} for row in cursor.fetchall()]

    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def get_metadata_stats() -> dict:
    """
    Get statistics about loaded metadata.

    Returns:
        dict with stats about metadata coverage
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Total records
        cursor.execute('SELECT COUNT(*) FROM trace_metadata')
        total = cursor.fetchone()[0]

        # Matched records
        cursor.execute('SELECT COUNT(*) FROM trace_metadata WHERE match_success = 1')
        matched = cursor.fetchone()[0]

        # Unique workspaces
        cursor.execute('SELECT COUNT(DISTINCT workspace_name) FROM trace_metadata WHERE workspace_name IS NOT NULL')
        unique_workspaces = cursor.fetchone()[0]

        # Unique patients
        cursor.execute('SELECT COUNT(DISTINCT patient_id) FROM trace_metadata WHERE patient_id IS NOT NULL')
        unique_patients = cursor.fetchone()[0]

        # By use case
        cursor.execute('''
            SELECT use_case, COUNT(*) as count, SUM(match_success) as matched
            FROM trace_metadata
            GROUP BY use_case
            ORDER BY count DESC
        ''')
        by_use_case = [
            {'use_case': row[0], 'count': row[1], 'matched': row[2]}
            for row in cursor.fetchall()
        ]

        return {
            'total_records': total,
            'matched_records': matched,
            'match_rate': (matched / total * 100) if total > 0 else 0,
            'unique_workspaces': unique_workspaces,
            'unique_patients': unique_patients,
            'by_use_case': by_use_case
        }

    except sqlite3.OperationalError:
        return {
            'total_records': 0,
            'matched_records': 0,
            'match_rate': 0,
            'unique_workspaces': 0,
            'unique_patients': 0,
            'by_use_case': []
        }
    finally:
        conn.close()


def metadata_table_exists() -> bool:
    """Check if the trace_metadata table exists and has data."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT COUNT(*) FROM trace_metadata')
        count = cursor.fetchone()[0]
        return count > 0
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    # CLI for loading metadata
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Load trace metadata from matched JSON files")
    parser.add_argument("--stats", action="store_true", help="Show metadata statistics")
    parser.add_argument("--load", action="store_true", help="Load/reload all metadata files")
    args = parser.parse_args()

    if args.load or not args.stats:
        print(f"\nLoading metadata from {DATA_DIR}...")
        result = load_all_metadata()
        print(f"\nLoading complete:")
        print(f"  Files found: {result['files_found']}")
        print(f"  Records loaded: {result['total_loaded']}")
        print(f"  Records skipped: {result['total_skipped']}")

    if args.stats or not args.load:
        print("\nMetadata Statistics:")
        stats = get_metadata_stats()
        print(f"  Total records: {stats['total_records']}")
        print(f"  Matched records: {stats['matched_records']} ({stats['match_rate']:.1f}%)")
        print(f"  Unique workspaces: {stats['unique_workspaces']}")
        print(f"  Unique patients: {stats['unique_patients']}")

        if stats['by_use_case']:
            print("\n  By use case:")
            for uc in stats['by_use_case']:
                print(f"    {uc['use_case']}: {uc['count']} ({uc['matched']} matched)")
