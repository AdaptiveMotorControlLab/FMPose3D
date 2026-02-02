#!/usr/bin/env python3
"""
Script to update file headers in the FMPose3D repository.

This script replaces the old header format with the new header format
across all Python files in the repository.
"""

import os
import sys
from pathlib import Path

# Define the old and new headers
OLD_HEADER = '''"""
FMPose: 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose: 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Accepted by IEEE Transactions on Multimedia (TMM), 2025.
"""'''

NEW_HEADER = '''"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose: 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""'''


def update_file_header(file_path):
    """
    Update the header in a single file.
    
    Args:
        file_path: Path to the file to update
        
    Returns:
        True if the file was updated, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if OLD_HEADER in content:
            new_content = content.replace(OLD_HEADER, NEW_HEADER)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def find_and_update_headers(root_dir):
    """
    Find and update all Python files with the old header.
    
    Args:
        root_dir: Root directory to search from
        
    Returns:
        List of files that were updated
    """
    root_path = Path(root_dir)
    updated_files = []
    
    # Find all Python files
    for py_file in root_path.rglob('*.py'):
        # Skip files in .git directory
        if '.git' in py_file.parts:
            continue
            
        if update_file_header(py_file):
            updated_files.append(py_file)
            print(f"✓ Updated: {py_file.relative_to(root_path)}")
    
    return updated_files


def main():
    """Main function to run the header update script."""
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = os.getcwd()
    
    print(f"Searching for files with old headers in: {root_dir}")
    print("-" * 60)
    
    updated_files = find_and_update_headers(root_dir)
    
    print("-" * 60)
    if updated_files:
        print(f"\n✓ Successfully updated {len(updated_files)} file(s):")
        for file_path in updated_files:
            print(f"  - {file_path}")
    else:
        print("\nNo files found with the old header.")
    
    return 0 if updated_files else 1


if __name__ == '__main__':
    sys.exit(main())
