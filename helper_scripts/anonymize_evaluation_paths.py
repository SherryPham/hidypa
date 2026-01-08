"""
Script to anonymize identifying paths in evaluation JSON files.

This script removes hardcoded HPC paths (account IDs, usernames, full paths) from
all summary.json files in the evaluation/ and evaluation_2_backup/ directories.

It replaces absolute paths like:
    "/fred/oz411/kpham/crypto-watermark/evaluation/..."
with relative paths like:
    "evaluation/..."

Usage:
    python helper_scripts/anonymize_evaluation_paths.py
"""

import json
import os
import re
from pathlib import Path


def anonymize_output_directory(output_dir: str) -> str:
    """
    Convert absolute path to relative path by removing HPC-specific prefixes.
    
    Examples:
        "/fred/oz411/kpham/crypto-watermark/evaluation/..." -> "evaluation/..."
        "/some/other/path/evaluation_2_backup/..." -> "evaluation_2_backup/..."
    """
    if not output_dir:
        return output_dir
    
    # Pattern to match common HPC path prefixes
    # Matches: /fred/oz411/kpham/crypto-watermark/ or any absolute path ending with the project
    patterns = [
        r'^/fred/oz\d+/[^/]+/crypto-watermark/',
        r'^/.*?/crypto-watermark/',
        r'^/.*?/Cryptographic-Watermarking-for-LLM/',
    ]
    
    for pattern in patterns:
        if re.match(pattern, output_dir):
            # Extract the relative part after the project directory
            match = re.search(pattern, output_dir)
            relative_part = output_dir[match.end():]
            return relative_part
    
    # If it's already a relative path or doesn't match patterns, return as-is
    # But check if it starts with evaluation or evaluation_2_backup
    if output_dir.startswith('evaluation') or output_dir.startswith('evaluation_2_backup'):
        return output_dir
    
    # If we can't determine, return None to indicate it should be removed
    return None


def anonymize_json_file(json_path: Path) -> bool:
    """
    Anonymize paths in a single JSON file.
    
    Returns:
        True if file was modified, False otherwise
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        
        # Anonymize output_directory field
        if 'output_directory' in data:
            original = data['output_directory']
            anonymized = anonymize_output_directory(original)
            
            if anonymized is None:
                # Remove the field if we can't anonymize it safely
                del data['output_directory']
                modified = True
                print(f"  Removed output_directory field (could not anonymize: {original})")
            elif anonymized != original:
                data['output_directory'] = anonymized
                modified = True
                print(f"  Anonymized: {original[:60]}... -> {anonymized}")
        
        # Check for any other fields that might contain paths
        # Recursively check nested dictionaries
        def check_nested_dict(obj, path=""):
            nonlocal modified
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and ('/fred/' in value or '/oz' in value):
                        # Try to anonymize any string that looks like a path
                        anonymized = anonymize_output_directory(value)
                        if anonymized and anonymized != value:
                            obj[key] = anonymized
                            modified = True
                            print(f"  Anonymized nested field {path}.{key}")
                    elif isinstance(value, (dict, list)):
                        check_nested_dict(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_nested_dict(item, f"{path}[{i}]" if path else f"[{i}]")
        
        check_nested_dict(data)
        
        if modified:
            # Write back the anonymized JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        
        return False
        
    except json.JSONDecodeError as e:
        print(f"  ERROR: Invalid JSON in {json_path}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: Failed to process {json_path}: {e}")
        return False


def main():
    """Main function to find and anonymize all evaluation JSON files."""
    # Get the project root (parent of helper_scripts)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Directories to search
    search_dirs = [
        project_root / 'evaluation',
        project_root / 'evaluation_2_backup',
    ]
    
    print("=" * 80)
    print("Anonymizing Evaluation JSON Files")
    print("=" * 80)
    print()
    
    total_files = 0
    modified_files = 0
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            print(f"Directory not found: {search_dir}")
            print()
            continue
        
        print(f"Searching in: {search_dir}")
        
        # Find all summary.json files
        json_files = list(search_dir.rglob('summary.json'))
        
        if not json_files:
            print(f"  No summary.json files found")
            print()
            continue
        
        print(f"  Found {len(json_files)} summary.json file(s)")
        print()
        
        for json_file in sorted(json_files):
            total_files += 1
            rel_path = json_file.relative_to(project_root)
            print(f"Processing: {rel_path}")
            
            if anonymize_json_file(json_file):
                modified_files += 1
                print(f"  [MODIFIED]")
            else:
                print(f"  [NO CHANGES]")
            print()
    
    print("=" * 80)
    print(f"Summary: {modified_files} of {total_files} files modified")
    print("=" * 80)


if __name__ == '__main__':
    main()

