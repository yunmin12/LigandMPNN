#!/usr/bin/env python3
# filepath: /home/yunmin/proj/LigandMPNN/dataprep/sair_structure_list.py
"""
Generate structure file lists from SAIR CSV based on entry_id ranges.
Creates text files with CIF file paths and maintains a TSV index of tarballs.

Usage:
  Single range: python sair_structure_list.py data.csv --range 0 10565
  Multiple ranges from file: python sair_structure_list.py data.csv --range_file ranges.txt
"""

import pandas as pd
import argparse
from pathlib import Path

def process_range(df, range_start, range_end, out_dir):
    """Process a single range and generate list file."""
    
    # Filter by range
    filtered = df[
        (df['entry_id'] >= range_start) & 
        (df['entry_id'] <= range_end)
    ]
    
    print(f"\nProcessing range [{range_start}, {range_end}]...")
    print(f"  Found {len(filtered)} entries")
    
    if len(filtered) == 0:
        print("  ⚠️  No entries found in this range, skipping.")
        return False
    
    # Generate file paths
    paths = [
        f"structures/sample_{row['entry_id']}_model_{row['index']}.cif" 
        for _, row in filtered.iterrows()
    ]
    paths = sorted(set(paths))  # Remove duplicates and sort
    
    # Output filenames
    list_dir = out_dir / "lists"
    list_dir.mkdir(parents=True, exist_ok=True)
    
    list_file = list_dir / f"{range_start}_to_{range_end}.txt"
    tsv_file = out_dir / "manifest.tsv"
    tarball_name = f"sair_structures_{range_start}_to_{range_end}.tar.gz"
    
    # Write list file
    with open(list_file, 'w') as f:
        f.write('\n'.join(paths) + '\n')
    
    print(f"  ✅ Generated {list_file.name} ({len(paths)} files)")
    
    # Read existing TSV entries to avoid duplicates
    existing_entries = set()
    if tsv_file.exists():
        with open(tsv_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                existing_entries.add(line.strip())
    
    # Prepare new entry
    new_entry = f"{tarball_name}\tlists/{list_file.name}"
    
    # Append to TSV if not duplicate
    if new_entry not in existing_entries:
        # Create header if file doesn't exist
        if not tsv_file.exists():
            with open(tsv_file, 'w') as f:
                f.write("tarball\tlist_file\n")
        
        # Append new entry
        with open(tsv_file, 'a') as f:
            f.write(f"{new_entry}\n")
        
        print(f"  ✅ Added to manifest.tsv")
    else:
        print(f"  ⚠️  Entry already in manifest.tsv, skipping")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Generate structure file lists from CSV based on entry_id ranges"
    )
    parser.add_argument(
        "csv_file", 
        help="Input CSV file (e.g., sair_pIC50_best_per_protein.csv)"
    )
    
    # Mutually exclusive: either single range or range file
    range_group = parser.add_mutually_exclusive_group(required=True)
    range_group.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=('START', 'END'),
        help="Single range: START END (e.g., --range 0 10565)"
    )
    range_group.add_argument(
        "--range_file",
        help="File with entry_ids (first line = start, last line = end)"
    )
    
    parser.add_argument(
        "--out_dir", 
        default=".", 
        help="Output directory (default: current directory)"
    )
    args = parser.parse_args()
    
    # Read CSV once
    print(f"Reading {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    
    if 'entry_id' not in df.columns or 'index' not in df.columns:
        raise ValueError("CSV must contain 'entry_id' and 'index' columns")
    
    print(f"Loaded {len(df)} rows")
    
    # Setup output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse ranges
    ranges = []
    
    if args.range:
        # Single range mode
        start, end = args.range
        if start > end:
            raise ValueError(f"Invalid range: start ({start}) > end ({end})")
        ranges.append((start, end))
        print(f"Processing single range: [{start}, {end}]")
    
    else:
        # Range file mode
        print(f"\nReading ranges from {args.range_file}...")
        with open(args.range_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not lines:
            raise ValueError("Range file is empty or contains only comments")
        
        # Parse ranges (format: "start end" per line)
        for line_num, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 2:
                print(f"  ⚠️  Skipping invalid line {line_num}: '{line}' (expected 'start end')")
                continue
            
            try:
                start = int(parts[0])
                end = int(parts[1])
                if start > end:
                    print(f"  ⚠️  Skipping line {line_num}: start > end ({start} > {end})")
                    continue
                ranges.append((start, end))
            except ValueError:
                print(f"  ⚠️  Skipping line {line_num}: invalid integers '{line}'")
                continue
        
        if not ranges:
            raise ValueError("No valid ranges found in range file")
        
        print(f"Found {len(ranges)} valid range(s)")
    
    # Process ranges
    print("\n" + "="*60)
    print("PROCESSING RANGES")
    print("="*60)
    
    successful = 0
    for start, end in ranges:
        if process_range(df, start, end, out_dir):
            successful += 1
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total ranges: {len(ranges)}")
    print(f"Successful: {successful}")
    print(f"Output directory: {out_dir}")
    print(f"Manifest file: {out_dir / 'manifest.tsv'}")

if __name__ == "__main__":
    main()