"""
Check which proteins still have both target and off-target data without error during LigandMPNN running.
Output example:
    ============================================================
    Coverage Analysis for Kd
    ============================================================

    Proteins with complete loss of target or off-target:
    ✓ All proteins have at least some target AND off-target data!

    Overall Statistics:
    Total PDBs expected: 96
    Total PDBs missing: 0
    Success rate: 100.0%

    ============================================================
    Coverage Analysis for IC50
    ============================================================

    Proteins with complete loss of target or off-target:
    ⚠️  Q9H2K2: tar=0.0/3.0, off=0.0/3.0

    Overall Statistics:
    Total PDBs expected: 300
    Total PDBs missing: 13
    Success rate: 95.7%
"""

import pandas as pd
from pathlib import Path

def check_protein_coverage(assay_type):
    """Check which proteins still have both target and off-target data"""
    
    csv_path = f'/scratch/yunmin/data/graph/bdb/clustered/bindingdb_{assay_type}_clustered_stage1.csv'
    lmpnn_in = f'/scratch/yunmin/data/graph/lmpnn/bdb_pdb/{assay_type}/lmpnn_in'
    
    df = pd.read_csv(csv_path)
    
    # Count expected PDBs per protein
    expected = df.groupby(['uniprot_id', 'target_type']).size().reset_index(name='expected_count')
    
    # Count actual PDBs copied
    actual = []
    for protein_dir in Path(lmpnn_in).iterdir():
        if not protein_dir.is_dir():
            continue
        protein_id = protein_dir.name
        
        for target_dir in protein_dir.iterdir():
            if target_dir.name not in ['tar', 'off']:
                continue
            target_type = 'target' if target_dir.name == 'tar' else 'off_target'
            
            pdb_count = sum(1 for _ in target_dir.rglob('*.pdb'))
            actual.append({
                'uniprot_id': protein_id,
                'target_type': target_type,
                'actual_count': pdb_count
            })
    
    actual_df = pd.DataFrame(actual)
    
    # Merge and compare
    comparison = expected.merge(actual_df, on=['uniprot_id', 'target_type'], how='left')
    comparison['actual_count'] = comparison['actual_count'].fillna(0).astype(int)
    comparison['missing'] = comparison['expected_count'] - comparison['actual_count']
    
    # Find proteins missing all target or all off-target
    print(f"\n{'='*60}")
    print(f"Coverage Analysis for {assay_type}")
    print(f"{'='*60}")
    
    pivot = comparison.pivot_table(
        index='uniprot_id',
        columns='target_type',
        values=['expected_count', 'actual_count'],
        fill_value=0
    )
    
    print(f"\nProteins with complete loss of target or off-target:")
    problematic = []
    
    for protein in pivot.index:
        tar_expected = pivot.loc[protein, ('expected_count', 'target')]
        tar_actual = pivot.loc[protein, ('actual_count', 'target')]
        off_expected = pivot.loc[protein, ('expected_count', 'off_target')]
        off_actual = pivot.loc[protein, ('actual_count', 'off_target')]
        
        if (tar_expected > 0 and tar_actual == 0) or (off_expected > 0 and off_actual == 0):
            print(f"  ⚠️  {protein}: tar={tar_actual}/{tar_expected}, off={off_actual}/{off_expected}")
            problematic.append(protein)
    
    if not problematic:
        print("  ✓ All proteins have at least some target AND off-target data!")
    
    # Overall stats
    total_missing = comparison['missing'].sum()
    total_expected = comparison['expected_count'].sum()
    
    print(f"\nOverall Statistics:")
    print(f"  Total PDBs expected: {total_expected}")
    print(f"  Total PDBs missing: {total_missing}")
    print(f"  Success rate: {100*(1-total_missing/total_expected):.1f}%")
    
    return problematic

# Run for both assay types
for assay in ['Kd', 'IC50']:
    problematic = check_protein_coverage(assay)
