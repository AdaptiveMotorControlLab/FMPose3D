#!/usr/bin/env python3
"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import os
import sys
from pathlib import Path

# Define the standard header for the project
STANDARD_HEADER = '''"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""'''

# Old headers that should be replaced
OLD_HEADERS = [
    '''"""
FMPose: 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose: 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Accepted by IEEE Transactions on Multimedia (TMM), 2025.
"""'''
]


def should_skip_file(file_path):
    """
    Determine if a file should be skipped for header addition.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if the file should be skipped, False otherwise
    """
    skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', 'env', '.tox', 'build', 'dist', '.eggs'}
    
    # Skip if in excluded directory
    for part in file_path.parts:
        if part in skip_dirs:
            return True
    
    # Skip __init__.py files that are typically minimal
    if file_path.name == '__init__.py':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Skip if __init__.py is very short (likely just imports)
            if len(content.strip()) < 50:
                return True
        except Exception:
            pass
    
    return False


def has_header(content):
    """
    Check if content already has the standard header.
    
    Args:
        content: File content to check
        
    Returns:
        True if the file has the standard header, False otherwise
    """
    return STANDARD_HEADER.strip() in content


def needs_header_update(content):
    """
    Check if content has an old header that needs updating.
    
    Args:
        content: File content to check
        
    Returns:
        Old header if found, None otherwise
    """
    for old_header in OLD_HEADERS:
        if old_header.strip() in content:
            return old_header
    return None


def add_or_update_header(file_path, check_only=False):
    """
    Add or update the header in a single file.
    
    Args:
        file_path: Path to the file to update
        check_only: If True, only check without modifying
        
    Returns:
        Tuple of (status, message) where status is 'ok', 'updated', 'added', or 'error'
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file already has the correct header
        if has_header(content):
            return ('ok', 'Already has correct header')
        
        # Check if file has an old header that needs replacing
        old_header = needs_header_update(content)
        if old_header:
            if not check_only:
                new_content = content.replace(old_header, STANDARD_HEADER)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            return ('updated', 'Replaced old header with standard header')
        
        # File has no header, add one
        # Skip adding header to files that start with shebang or are very short
        lines = content.split('\n')
        if content.strip() and len(content.strip()) > 10:
            if not check_only:
                # Handle special cases for header placement
                new_lines = []
                insert_index = 0
                
                # If file starts with shebang, keep it at the top
                if lines[0].startswith('#!'):
                    new_lines.append(lines[0])
                    insert_index = 1
                
                # Check for 'from __future__' imports which must be very early
                # Find the first non-comment, non-shebang, non-empty line
                future_import_index = None
                for i in range(insert_index, min(len(lines), 10)):
                    line = lines[i].strip()
                    if line.startswith('from __future__'):
                        future_import_index = i
                        break
                    elif line and not line.startswith('#'):
                        # Found a non-comment line that isn't a future import
                        break
                
                if future_import_index is not None:
                    # If there's a from __future__ import, add header AFTER it
                    new_lines.extend(lines[insert_index:future_import_index+1])
                    new_lines.append(STANDARD_HEADER)
                    new_lines.append('')
                    new_lines.extend(lines[future_import_index+1:])
                else:
                    # Otherwise, add header at the beginning (after shebang if present)
                    new_lines.append(STANDARD_HEADER)
                    new_lines.append('')
                    new_lines.extend(lines[insert_index:])
                
                new_content = '\n'.join(new_lines)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            return ('added', 'Added standard header')
        
        return ('ok', 'Skipped (file too short or empty)')
        
    except Exception as e:
        return ('error', f"Error processing file: {e}")


def find_and_process_headers(root_dir, check_only=False):
    """
    Find and process all Python files.
    
    Args:
        root_dir: Root directory to search from
        check_only: If True, only check without modifying files
        
    Returns:
        Dictionary with statistics about processed files
    """
    root_path = Path(root_dir)
    stats = {
        'ok': [],
        'updated': [],
        'added': [],
        'error': []
    }
    
    # Find all Python files
    for py_file in root_path.rglob('*.py'):
        # Skip files that should not be processed
        if should_skip_file(py_file):
            continue
            
        status, message = add_or_update_header(py_file, check_only)
        stats[status].append((py_file, message))
        
        if status in ['updated', 'added']:
            rel_path = py_file.relative_to(root_path)
            print(f"{'[CHECK]' if check_only else '✓'} {rel_path}: {message}")
        elif status == 'error':
            rel_path = py_file.relative_to(root_path)
            print(f"✗ {rel_path}: {message}")
    
    return stats


def main():
    """Main function to run the header update script."""
    check_only = '--check' in sys.argv
    
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        root_dir = sys.argv[1]
    else:
        root_dir = os.getcwd()
    
    mode = "Checking" if check_only else "Processing"
    print(f"{mode} files for headers in: {root_dir}")
    print("-" * 60)
    
    stats = find_and_process_headers(root_dir, check_only)
    
    print("-" * 60)
    
    # Print summary
    total_changes = len(stats['updated']) + len(stats['added'])
    
    if check_only:
        if total_changes > 0:
            print(f"\n⚠ Found {total_changes} file(s) needing header updates:")
            for file_path, msg in stats['updated']:
                print(f"  - {file_path.relative_to(root_dir)}: {msg}")
            for file_path, msg in stats['added']:
                print(f"  - {file_path.relative_to(root_dir)}: {msg}")
            return 1
        else:
            print("\n✓ All Python files have correct headers!")
            return 0
    else:
        if total_changes > 0:
            print(f"\n✓ Successfully processed {total_changes} file(s):")
            if stats['updated']:
                print(f"  - Updated: {len(stats['updated'])} file(s)")
            if stats['added']:
                print(f"  - Added headers: {len(stats['added'])} file(s)")
        else:
            print("\n✓ No files needed header updates.")
        
        if stats['error']:
            print(f"\n✗ Errors: {len(stats['error'])} file(s)")
            for file_path, msg in stats['error']:
                print(f"  - {file_path.relative_to(root_dir)}: {msg}")
            return 1
        
        return 0


if __name__ == '__main__':
    sys.exit(main())
