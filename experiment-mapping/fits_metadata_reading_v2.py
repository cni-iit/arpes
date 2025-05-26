#!/usr/bin/env python3
"""
FITS Metadata Parser - Enhanced Version
Extracts specified metadata from FITS files and exports to CSV format.
Allows custom key selection and renaming.
Handles two consecutive metadata blocks ending with 'END'.
"""

import os
import csv
import glob
from pathlib import Path
import re

# Configuration: Specify which keys to extract and how to rename them
# Format: 'ORIGINAL_KEY': 'NEW_COLUMN_NAME'
# Set to None or empty dict to extract all keys (original behavior)
SELECTED_KEYS = {
    # Example configuration - modify these based on your FITS files
    'FILENAME': 'Filename',
    'PMOTOR0': 'Stage_X',
    'PMOTOR1': 'Stage_Y',
    'PMOTOR3': 'Stage_Theta',
    'PMOTOR4': 'Stage_Phi',
    'LMOTOR0': 'Logical_stage_X',
    'LMOTOR1': 'Logical_stage_Y',
    'LMOTOR10': 'Capillary_X',
    'LMOTOR11': 'Capillary_Y',
    'LWLVNM': 'Measurement_type',
    'ST_0_0': 'Scan_start_x',
    'EN_0_0': 'Scan_end_x',
    'ST_0_1': 'Scan_start_y',
    'EN_0_1': 'Scan_end_y',
    # Add more mappings as needed
}

# Alternative: Set to None to extract ALL keys (original behavior)
# SELECTED_KEYS = None

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

def filter_and_rename_metadata(metadata, selected_keys=None):
    """
    Filter metadata to only include selected keys and rename them.
    
    Args:
        metadata: Dictionary of all metadata
        selected_keys: Dictionary mapping original keys to new names, or None for all keys
    
    Returns:
        Dictionary with filtered and renamed metadata
    """
    if selected_keys is None:
        return metadata
    
    filtered_metadata = {}
    
    for original_key, new_name in selected_keys.items():
        if original_key in metadata:
            filtered_metadata[new_name] = metadata[original_key]
        else:
            # Key not found, add empty value to maintain consistent CSV structure
            filtered_metadata[new_name] = ''
    
    return filtered_metadata

def print_available_keys(all_metadata):
    """
    Print all available keys found in the FITS files to help with configuration.
    """
    all_keys = set()
    for metadata in all_metadata:
        all_keys.update(metadata.keys())
    
    print("\nAvailable metadata keys found in FITS files:")
    print("=" * 50)
    for key in sorted(all_keys):
        if key != 'FILENAME':
            print(f"  {key}")
    print("=" * 50)
    print("Copy and modify the SELECTED_KEYS dictionary in the script to choose which keys to extract.")

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
    
    for fits_file in fits_files:
        print(f"\nProcessing {os.path.basename(fits_file)}...")
        metadata = parse_fits_file(fits_file)
        
        if metadata:
            # Add filename to metadata
            metadata['FILENAME'] = os.path.basename(fits_file)
            all_metadata.append(metadata)
        else:
            print(f"  Failed to parse {fits_file}")
    
    if not all_metadata:
        print("No metadata extracted from any files.")
        return
    
    # Show available keys if no selection is configured
    if SELECTED_KEYS is None or len(SELECTED_KEYS) == 0:
        print_available_keys(all_metadata)
        print("\nExtracting ALL metadata keys...")
        # Use original behavior
        all_keys = set()
        for metadata in all_metadata:
            all_keys.update(metadata.keys())
        
        ordered_keys = []
        for metadata in all_metadata:
            for key in metadata.keys():
                if key not in ordered_keys and key != 'FILENAME':
                    ordered_keys.append(key)
        
        final_keys = ['FILENAME'] + ordered_keys
        filtered_metadata_list = all_metadata
    else:
        print(f"\nUsing custom key selection: {len(SELECTED_KEYS)} keys specified")
        print("Selected keys mapping:")
        for orig, new in SELECTED_KEYS.items():
            print(f"  {orig} → {new}")
        
        # Filter and rename metadata
        filtered_metadata_list = []
        for metadata in all_metadata:
            filtered = filter_and_rename_metadata(metadata, SELECTED_KEYS)
            filtered['FILENAME'] = metadata['FILENAME']  # Always keep filename
            filtered_metadata_list.append(filtered)
        
        # Create column order: filename first, then renamed keys in order
        final_keys = ['FILENAME'] + list(SELECTED_KEYS.values())
    
    # Write to CSV
    output_file = 'fits_metadata.csv'
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=final_keys)
            writer.writeheader()
            
            for metadata in filtered_metadata_list:
                # Handle list values (convert to string)
                row = {}
                for key in final_keys:
                    value = metadata.get(key, '')
                    if isinstance(value, list):
                        value = '; '.join(str(v) for v in value)
                    row[key] = value
                writer.writerow(row)
        
        print(f"\nMetadata successfully exported to '{output_file}'")
        print(f"Processed {len(filtered_metadata_list)} files with {len(final_keys)} columns")
        
        if SELECTED_KEYS:
            # Show summary of found vs missing keys
            found_keys = set()
            for metadata in all_metadata:
                found_keys.update(metadata.keys())
            
            missing_keys = set(SELECTED_KEYS.keys()) - found_keys
            if missing_keys:
                print(f"\nWarning: {len(missing_keys)} selected keys not found in any FITS files:")
                for key in sorted(missing_keys):
                    print(f"  - {key}")
        
    except Exception as e:
        print(f"Error writing CSV file: {e}")

if __name__ == "__main__":
    main()