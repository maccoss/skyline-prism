#!/usr/bin/env python3
"""
Create a subset of the large CSV files containing ~10% of peptides.
Uses DuckDB for efficient streaming through large files.
"""
import duckdb
from pathlib import Path
import sys

input_dir = Path("example-files")
output_dir = Path("example-files/subset")
output_dir.mkdir(exist_ok=True)

# Connect to DuckDB (in-memory, but uses streaming)
con = duckdb.connect()

# Files to process
csv_files = [
    "2025-IRType-Plasma-PRISM-Plate1.csv",
    "2025-IRType-Plasma-PRISM-Plate2.csv", 
    "2025-IRType-Plasma-PRISM-Plate3.csv",
]

# First, get all unique peptides across all files
print("Collecting unique peptides from all files...")
print("  (This may take a few minutes for 47GB of data)")
sys.stdout.flush()

peptide_query = " UNION ALL ".join([
    f"SELECT DISTINCT \"Peptide Modified Sequence Unimod Ids\" as peptide FROM read_csv_auto('{input_dir / f}')"
    for f in csv_files
])

peptides_df = con.execute(f"""
    SELECT DISTINCT peptide 
    FROM ({peptide_query})
""").fetchdf()

n_total = len(peptides_df)
print(f"Total unique peptides: {n_total:,}")

# Select ~10% using hash
# DuckDB has hash() function
print("Selecting 10% subset...")
con.register('peptides_table', peptides_df)

selected = con.execute("""
    SELECT peptide 
    FROM peptides_table 
    WHERE hash(peptide) % 10 = 0
""").fetchdf()

n_selected = len(selected)
print(f"Selected {n_selected:,} peptides ({n_selected/n_total*100:.1f}%)")

# Register selected peptides for filtering
con.register('selected_peptides', selected)

# Now create subset files
print("\nCreating subset files...")
for csv_file in csv_files:
    input_path = input_dir / csv_file
    output_path = output_dir / csv_file.replace('.csv', '_subset.csv')
    
    print(f"  Processing {csv_file}...", end=" ", flush=True)
    
    # Use DuckDB to filter and write
    con.execute(f"""
        COPY (
            SELECT * FROM read_csv_auto('{input_path}')
            WHERE "Peptide Modified Sequence Unimod Ids" IN (SELECT peptide FROM selected_peptides)
        ) TO '{output_path}' (HEADER, DELIMITER ',')
    """)
    
    # Get row count
    result = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{output_path}')").fetchone()
    print(f"{result[0]:,} rows")

print("\nDone! Subset files created in:", output_dir)
print("\nTo run PRISM on the subset:")
print(f"  prism run -i {output_dir}/*_subset.csv -o example-files/output_subset")
