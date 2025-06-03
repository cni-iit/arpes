#!/usr/bin/env python3
"""
FITS Metadata Parser
Extracts metadata from FITS files and exports to CSV format.
Handles two consecutive metadata blocks ending with 'END'.
"""

import os
import csv
import glob
from pathlib import Path
import re

def parse_fits_metadata_block(file_handle):
    """
    Parse a single FITS metadata block (80-character records ending with 'END').
    Returns a dictionary of key-value pairs.
    """
    metadata = {}
    
    while True:
        # Read 80-character record
        record = file_handle.read(80)
        if len(record) < 80:
            break
            
        # Convert bytes to string if necessary
        if isinstance(record, bytes):
            record = record.decode('ascii', errors='ignore')
        
        # Check for END keyword
        if record.startswith('END'):
            break
            
        # Parse key-value pairs (FITS format: KEYWORD = VALUE / COMMENT)
        if '=' in record:
            # Split at first '=' sign
            key_part, value_part = record.split('=', 1)
            key = key_part.strip()
            
            # Extract value (may include comment after '/')
            value = value_part.strip()
            if '/' in value:
                value = value.split('/', 1)[0].strip()
            
            # Remove quotes from string values
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1].strip()
            
            # Try to convert to appropriate type
            try:
                # Try integer first
                if '.' not in value and value.replace('-', '').replace('+', '').isdigit():
                    value = int(value)
                # Try float
                elif value.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit():
                    value = float(value)
                # Handle boolean
                elif value.upper() in ['T', 'F']:
                    value = value.upper() == 'T'
            except ValueError:
                pass  # Keep as string
                
            metadata[key] = value
        
        # Handle comment-only records (starting with COMMENT, HISTORY, etc.)
        elif record.startswith(('COMMENT', 'HISTORY', 'HIERARCH')):
            key_match = re.match(r'^(\w+)\s+(.*)', record)
            if key_match:
                key = key_match.group(1)
                value = key_match.group(2).strip()
                # For multiple comments, create a list
                if key in metadata:
                    if not isinstance(metadata[key], list):
                        metadata[key] = [metadata[key]]
                    metadata[key].append(value)
                else:
                    metadata[key] = value
    
    return metadata

def parse_fits_file(filepath):
    """
    Parse a FITS file and extract metadata from two consecutive blocks.
    Returns a dictionary with combined metadata from both blocks.
    """
    metadata = {}
    
    try:
        with open(filepath, 'rb') as f:
            # Parse first metadata block
            block1 = parse_fits_metadata_block(f)
            
            # Add block identifier to keys to avoid conflicts
            for key, value in block1.items():
                metadata[f"{key}"] = value
            
            # Parse second metadata block
            block2 = parse_fits_metadata_block(f)
            
            # Add second block with identifier
            for key, value in block2.items():
                metadata[f"{key}"] = value
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None
    
    return metadata

def main():
    """
    Main function to process all FITS files in the current directory.
    """
    # Get current directory
    current_dir = Path('.')
    
    # Find all FITS files (common extensions)
    fits_patterns = ['*.fits', '*.fit', '*.fts']
    fits_files = []
    
    for pattern in fits_patterns:
        fits_files.extend(glob.glob(str(current_dir / pattern)))
    
    if not fits_files:
        print("No FITS files found in the current directory.")
        return
    
    print(f"Found {len(fits_files)} FITS files:")
    for f in fits_files:
        print(f"  - {os.path.basename(f)}")
    
    # Parse all files and collect metadata
    all_metadata = []
    all_keys = set()
    
    for fits_file in fits_files:
        print(f"\nProcessing {os.path.basename(fits_file)}...")
        metadata = parse_fits_file(fits_file)
        
        if metadata:
            # Add filename to metadata
            metadata['FILENAME'] = os.path.basename(fits_file)
            all_metadata.append(metadata)
            all_keys.update(metadata.keys())
        else:
            print(f"  Failed to parse {fits_file}")
    
    if not all_metadata:
        print("No metadata extracted from any files.")
        return
    
    # Preserve order of appearance instead of alphabetical sorting
    # Collect keys in the order they first appear across all files
    ordered_keys = []
    for metadata in all_metadata:
        for key in metadata.keys():
            if key not in ordered_keys and key != 'FILENAME':
                ordered_keys.append(key)
    
    # Move FILENAME to first column
    final_keys = ['FILENAME'] + ordered_keys
    
    # Write to CSV
    output_file = 'fits_metadata.csv'
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=final_keys)
            writer.writeheader()
            
            for metadata in all_metadata:
                # Handle list values (convert to string)
                row = {}
                for key in final_keys:
                    value = metadata.get(key, '')
                    if isinstance(value, list):
                        value = '; '.join(str(v) for v in value)
                    row[key] = value
                writer.writerow(row)
        
        print(f"\nMetadata successfully exported to '{output_file}'")
        print(f"Processed {len(all_metadata)} files with {len(final_keys)} unique metadata keys")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")

if __name__ == "__main__":
    main()