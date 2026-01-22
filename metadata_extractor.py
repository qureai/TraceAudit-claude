"""
Metadata Extractor for LLM Traces

This module extracts identifiable information (patient_id, report_hash) from traces
that can be used to match against Django models in a separate environment.

Part 1: Run here to extract data from traces
Part 2: Run in Django sandbox to match against DB (code provided at bottom)
"""

import json
import hashlib
import re
import sqlite3
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any
from config import TRACES_DB, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExtractedMetadata:
    """Metadata extracted from a trace for DB matching."""
    element_id: str
    trace_id: str
    use_case: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    workspace_name: Optional[str] = None  # Extracted from trace (for validation)
    report_hash: Optional[str] = None
    report_preview: Optional[str] = None  # First 100 chars for debugging
    extraction_success: bool = False
    extraction_error: Optional[str] = None


class UseCaseExtractor:
    """Base class for use-case specific extractors."""

    use_case_name: str = "base"

    def extract(self, trace_data: Dict[str, Any]) -> ExtractedMetadata:
        """Extract metadata from a trace. Override in subclasses."""
        raise NotImplementedError

    def get_user_message(self, trace_data: Dict[str, Any]) -> Optional[str]:
        """Get the user message content from trace."""
        messages = trace_data.get('request', {}).get('messages', [])
        for msg in messages:
            if msg.get('role') == 'user':
                return msg.get('content', '')
        return None

    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent hashing."""
        return text.strip().replace('\r\n', '\n').replace('\r', '\n')

    def compute_hash(self, text: str) -> str:
        """Compute MD5 hash of normalized text (first 16 chars)."""
        normalized = self.normalize_text(text)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]


class RadReportDataExtractionExtractor(UseCaseExtractor):
    """
    Extractor for worklist_rad_report_data_extraction use case.

    Expected user message format:
    **Patient Data:**
    Patient Id: XXXXX, Patient Name: YYYY, Patient Age: ZZ, Patient Gender: G

    **Radiology Report:**
    ...report text...

    **Available Tags:**
    [...]
    """

    use_case_name = "worklist_rad_report_data_extraction"

    # Regex patterns
    PATIENT_ID_PATTERN = re.compile(r'Patient Id:\s*([^,\n]+)')
    PATIENT_NAME_PATTERN = re.compile(r'Patient Name:\s*([^,\n]+)')
    REPORT_PATTERN = re.compile(
        r'\*\*Radiology Report:\*\*\s*(.*?)\*\*Available Tags:',
        re.DOTALL
    )

    def extract(self, trace_data: Dict[str, Any]) -> ExtractedMetadata:
        """Extract patient_id and report_hash from rad report trace."""
        element_id = trace_data.get('id', '')
        trace_id = trace_data.get('trace_id', '')

        metadata = ExtractedMetadata(
            element_id=element_id,
            trace_id=trace_id,
            use_case=self.use_case_name
        )

        try:
            user_message = self.get_user_message(trace_data)
            if not user_message:
                metadata.extraction_error = "No user message found"
                return metadata

            # Extract patient_id
            patient_id_match = self.PATIENT_ID_PATTERN.search(user_message)
            if patient_id_match:
                metadata.patient_id = patient_id_match.group(1).strip()
            else:
                metadata.extraction_error = "Patient ID not found in message"
                return metadata

            # Extract patient_name (optional, for debugging)
            patient_name_match = self.PATIENT_NAME_PATTERN.search(user_message)
            if patient_name_match:
                metadata.patient_name = patient_name_match.group(1).strip()

            # Extract report text
            report_match = self.REPORT_PATTERN.search(user_message)
            if report_match:
                report_text = report_match.group(1).strip()
                metadata.report_hash = self.compute_hash(report_text)
                metadata.report_preview = report_text[:100]
                metadata.extraction_success = True
            else:
                metadata.extraction_error = "Report text not found in message"
                return metadata

        except Exception as e:
            metadata.extraction_error = f"Extraction failed: {str(e)}"

        return metadata


class WorklistTimelineExtractor(UseCaseExtractor):
    """
    Base extractor for worklist use cases with PATIENT TIMELINE format.

    Expected user message format:
    WORKSPACE INFORMATION:
      Workspace Name: {workspace_name}'s Workspace

    ================================================================================

    PATIENT TIMELINE - {patient_id}
    Patient Name: {patient_name}
    Age: {age}
    Gender: {gender}
    ...
    """

    # Regex patterns for timeline format
    PATIENT_ID_PATTERN = re.compile(r'PATIENT TIMELINE - ([^\n]+)')
    PATIENT_NAME_PATTERN = re.compile(r'Patient Name: ([^\n]+)')
    WORKSPACE_NAME_PATTERN = re.compile(r"Workspace Name: ([^']+)'s Workspace")

    def extract(self, trace_data: Dict[str, Any]) -> ExtractedMetadata:
        """Extract patient_id, patient_name, and workspace_name from timeline trace."""
        element_id = trace_data.get('id', '')
        trace_id = trace_data.get('trace_id', '')

        metadata = ExtractedMetadata(
            element_id=element_id,
            trace_id=trace_id,
            use_case=self.use_case_name
        )

        try:
            user_message = self.get_user_message(trace_data)
            if not user_message:
                metadata.extraction_error = "No user message found"
                return metadata

            # Extract patient_id
            patient_id_match = self.PATIENT_ID_PATTERN.search(user_message)
            if patient_id_match:
                metadata.patient_id = patient_id_match.group(1).strip()
            else:
                metadata.extraction_error = "Patient ID not found in PATIENT TIMELINE"
                return metadata

            # Extract patient_name
            patient_name_match = self.PATIENT_NAME_PATTERN.search(user_message)
            if patient_name_match:
                metadata.patient_name = patient_name_match.group(1).strip()
            else:
                metadata.extraction_error = "Patient Name not found"
                return metadata

            # Extract workspace_name (optional - for validation)
            workspace_match = self.WORKSPACE_NAME_PATTERN.search(user_message)
            if workspace_match:
                metadata.workspace_name = workspace_match.group(1).strip()

            metadata.extraction_success = True

        except Exception as e:
            metadata.extraction_error = f"Extraction failed: {str(e)}"

        return metadata


class SuggestNextStepsExtractor(WorklistTimelineExtractor):
    """Extractor for worklist_suggest_next_steps use case."""
    use_case_name = "worklist_suggest_next_steps"


class PatientSummaryExtractor(WorklistTimelineExtractor):
    """Extractor for worklist_patient_summary use case."""
    use_case_name = "worklist_patient_summary"


# Registry of extractors by use case
EXTRACTORS: Dict[str, UseCaseExtractor] = {
    "worklist_rad_report_data_extraction": RadReportDataExtractionExtractor(),
    "worklist_suggest_next_steps": SuggestNextStepsExtractor(),
    "worklist_patient_summary": PatientSummaryExtractor(),
}


def get_traces_for_use_case(use_case_name: str, db_path: Path = TRACES_DB) -> List[Dict[str, Any]]:
    """Get all raw traces for a specific use case from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get prompt hash for this use case
    cursor.execute("""
        SELECT prompt_hash FROM prompts WHERE use_case_name = ?
    """, (use_case_name,))

    prompt_hashes = [row[0] for row in cursor.fetchall()]

    if not prompt_hashes:
        logger.warning(f"No prompt hash found for use case: {use_case_name}")
        conn.close()
        return []

    # Get all traces with these prompt hashes
    placeholders = ','.join(['?' for _ in prompt_hashes])
    cursor.execute(f"""
        SELECT r.data
        FROM raw_traces r
        JOIN traces t ON r.element_id = t.element_id
        WHERE t.system_prompt_hash IN ({placeholders})
    """, prompt_hashes)

    traces = []
    for row in cursor.fetchall():
        try:
            traces.append(json.loads(row[0]))
        except json.JSONDecodeError:
            continue

    conn.close()
    return traces


def extract_metadata_for_use_case(use_case_name: str) -> List[ExtractedMetadata]:
    """Extract metadata from all traces for a use case."""

    if use_case_name not in EXTRACTORS:
        raise ValueError(f"No extractor registered for use case: {use_case_name}")

    extractor = EXTRACTORS[use_case_name]
    traces = get_traces_for_use_case(use_case_name)

    logger.info(f"Found {len(traces)} traces for use case: {use_case_name}")

    results = []
    success_count = 0

    for trace in traces:
        metadata = extractor.extract(trace)
        results.append(metadata)
        if metadata.extraction_success:
            success_count += 1

    logger.info(f"Extraction complete: {success_count}/{len(traces)} successful")

    return results


def save_extracted_metadata(
    metadata_list: List[ExtractedMetadata],
    output_file: Path
) -> None:
    """Save extracted metadata to JSON file for use in Django sandbox."""
    data = [asdict(m) for m in metadata_list]

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(data)} records to {output_file}")


def extract_for_use_case(use_case: str) -> None:
    """Extract metadata for a specific use case and save to JSON."""
    output_file = OUTPUT_DIR / f"extracted_metadata_{use_case}.json"

    logger.info(f"Extracting metadata for use case: {use_case}")

    # Extract metadata from traces
    metadata_list = extract_metadata_for_use_case(use_case)

    # Save to JSON
    save_extracted_metadata(metadata_list, output_file)

    # Print summary
    success = sum(1 for m in metadata_list if m.extraction_success)
    failed = len(metadata_list) - success

    print(f"\n{'='*60}")
    print(f"Extraction Summary for: {use_case}")
    print(f"{'='*60}")
    print(f"Total traces:     {len(metadata_list)}")
    print(f"Successful:       {success}")
    print(f"Failed:           {failed}")
    print(f"Output file:      {output_file}")
    print(f"{'='*60}")

    # Show sample of extracted data
    if metadata_list:
        print("\nSample extracted record:")
        sample = next((m for m in metadata_list if m.extraction_success), metadata_list[0])
        print(json.dumps(asdict(sample), indent=2))

    # Show failure reasons if any
    if failed > 0:
        print("\nFailure reasons:")
        errors = {}
        for m in metadata_list:
            if not m.extraction_success and m.extraction_error:
                errors[m.extraction_error] = errors.get(m.extraction_error, 0) + 1
        for error, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {count:5d} - {error}")


def main():
    """Main entry point - extract metadata for specified use case(s)."""
    import sys

    if len(sys.argv) > 1:
        use_cases = sys.argv[1:]
    else:
        # Default: show available use cases
        print("Usage: python metadata_extractor.py <use_case> [use_case2 ...]")
        print("\nAvailable use cases:")
        for uc in EXTRACTORS.keys():
            print(f"  - {uc}")
        print("\nExample: python metadata_extractor.py worklist_suggest_next_steps")
        return

    for use_case in use_cases:
        if use_case not in EXTRACTORS:
            print(f"Error: Unknown use case '{use_case}'")
            print(f"Available: {', '.join(EXTRACTORS.keys())}")
            continue

        extract_for_use_case(use_case)
        print()  # Blank line between use cases


if __name__ == "__main__":
    main()


# =============================================================================
# PART 2: Django Sandbox Code
# =============================================================================
# Copy the code below and run it in your Django environment
# =============================================================================

DJANGO_SANDBOX_CODE = '''
"""
Django Sandbox Script - Run this in your Django environment

This script takes the extracted metadata JSON and matches it against
Django models to get workspace_id, account_id, and patient_pk.

Matching strategy: patient_id + patient_name (unique enough, no report hashing needed)

Usage:
    1. Copy extracted_metadata_worklist_rad_report_data_extraction.json to Django env
    2. Run: python manage.py shell < match_metadata.py
    Or import and call match_traces_to_db()
"""

import json
from typing import Dict, List, Optional

# Django imports - adjust paths as needed
from portal_manager.models import Patient


def normalize_name(name: str) -> str:
    """Normalize patient name for matching."""
    if not name:
        return ""
    return name.strip().upper()


def match_traces_to_db(
    extracted_metadata_path: str,
    output_path: str = "matched_metadata.json"
) -> List[dict]:
    """
    Match extracted trace metadata to Django DB records.

    Uses patient_id + patient_name for matching (unique enough across workspaces).

    Args:
        extracted_metadata_path: Path to JSON from Part 1
        output_path: Where to save matched results

    Returns:
        List of matched records with DB info populated
    """

    # Load extracted metadata
    with open(extracted_metadata_path, 'r') as f:
        extracted_data = json.load(f)

    print(f"Loaded {len(extracted_data)} extracted records")

    # Get unique patient_ids to query
    unique_patient_ids = set()
    for record in extracted_data:
        if record.get('patient_id') and record.get('extraction_success'):
            unique_patient_ids.add(record['patient_id'])

    print(f"Found {len(unique_patient_ids)} unique patient_ids to query")

    # Query patients for these patient_ids only
    patients = Patient.objects.filter(
        patient_id__in=list(unique_patient_ids)
    ).select_related('source')

    print(f"Found {patients.count()} patients in DB")

    # Build lookup: (patient_id, normalized_name) -> patient
    patient_lookup: Dict[tuple, Patient] = {}
    for p in patients:
        key = (p.patient_id, normalize_name(p.name))
        patient_lookup[key] = p

    print(f"Built lookup with {len(patient_lookup)} unique (patient_id, name) pairs")

    # Match traces to DB records
    results: List[dict] = []
    match_count = 0

    for record in extracted_data:
        # Copy original fields
        matched = {
            'element_id': record['element_id'],
            'trace_id': record['trace_id'],
            'use_case': record['use_case'],
            'patient_id': record.get('patient_id'),
            'patient_name': record.get('patient_name'),
            # DB fields to populate
            'patient_pk': None,
            'workspace_id': None,
            'workspace_name': None,
            'match_success': False,
            'match_error': None,
        }

        if not record.get('extraction_success'):
            matched['match_error'] = f"Extraction failed: {record.get('extraction_error')}"
            results.append(matched)
            continue

        # Try to match using patient_id + patient_name
        key = (record['patient_id'], normalize_name(record.get('patient_name', '')))

        if key in patient_lookup:
            p = patient_lookup[key]
            matched['patient_pk'] = p.pk
            matched['workspace_id'] = p.source_id
            matched['workspace_name'] = p.source.name if p.source else None
            matched['match_success'] = True
            match_count += 1
        else:
            # Fallback: try matching just patient_id (might get multiple)
            fallback_matches = [p for k, p in patient_lookup.items() if k[0] == record['patient_id']]
            if len(fallback_matches) == 1:
                p = fallback_matches[0]
                matched['patient_pk'] = p.pk
                matched['workspace_id'] = p.source_id
                matched['workspace_name'] = p.source.name if p.source else None
                matched['match_success'] = True
                matched['match_error'] = "Matched by patient_id only (name mismatch)"
                match_count += 1
            elif len(fallback_matches) > 1:
                matched['match_error'] = f"Multiple patients with same ID ({len(fallback_matches)}), name mismatch"
            else:
                matched['match_error'] = "No matching patient found in DB"

        results.append(matched)

    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\\nMatching complete:")
    print(f"  Total records:  {len(results)}")
    print(f"  Matched:        {match_count}")
    print(f"  Unmatched:      {len(results) - match_count}")
    print(f"  Output saved:   {output_path}")

    # Show unmatched breakdown
    unmatched = [r for r in results if not r['match_success']]
    if unmatched:
        print(f"\\nUnmatched reasons:")
        errors = {}
        for r in unmatched:
            err = r.get('match_error', 'Unknown')
            errors[err] = errors.get(err, 0) + 1
        for err, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {count:5d} - {err}")

    return results


# Run if executed directly
if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "extracted_metadata_worklist_rad_report_data_extraction.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "matched_metadata.json"

    match_traces_to_db(input_file, output_file)
'''

# To print the Django code for copying:
def print_django_code():
    print(DJANGO_SANDBOX_CODE)
