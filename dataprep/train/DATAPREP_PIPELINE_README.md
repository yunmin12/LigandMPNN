# Training Data Preparation Pipeline

Complete pipeline for preparing protein-ligand docking data for LigandMPNN training.

## Pipeline Stages

1. **Stage 1: Save Ligands** (`stage1_save_ligands.py`)
   - Download/extract target and off-target ligand files
   - Sources: PDB, PubChem, or generate from SMILES
   - Output: Ligand files in SDF/PDB format

2. **Stage 2: Prepare Target PDB** (`stage2_prep_target.py`)
   - Download target PDB structure
   - Extract and standardize ligand (Chain Z, ResID 1)
   - Remove waters and non-target HETATMs
   - Apply sequence blurring (convert all residues to ALA, backbone only)
   - Output: Standardized and blurred PDB files

3. **Stage 3: Generate Configs** (`stage3_generate_configs.py`)
   - Generate Vina configuration files for each complex
   - Define docking box parameters
   - Output: Individual YAML config files

4. **Stage 4: Vina Preparation** (uses `vina_prep.py`)
   - Convert PDB to PDBQT format
   - Prepare receptor and ligand for docking
   - Output: PDBQT files ready for Vina

5. **Stage 5: Vina Docking** (TODO)
   - Run AutoDock Vina
   - Generate multiple poses per complex

6. **Stage 6: Post-processing** (TODO)
   - Filter and rank poses
   - Select top candidates

## Usage

### Run Complete Pipeline

```bash
python pipeline.py \
  --csv /path/to/input.csv \
  --base_dir /scratch/data/training_run \
  --template config.yaml
```

### Run Specific Stages

```bash
# Run stages 1-3 only
python pipeline.py \
  --csv /path/to/input.csv \
  --base_dir /scratch/data/training_run \
  --template config.yaml \
  --start 1 --end 3
```

### Generate Progress Report

```bash
python pipeline.py \
  --csv /path/to/input.csv \
  --base_dir /scratch/data/training_run \
  --template config.yaml \
  --report-only
```

## Input CSV Format

Required columns:
- `protein_key`: Unique protein identifier
- `target_het_id`: Target ligand HET ID (3-letter code)
- `ligand_het_id`: Off-target ligand HET ID
- `valid_pdb_id`: PDB ID for structure download
- `ligand_smiles`: (optional) SMILES string for ligand generation
- `target_ligand_path`: (optional) Pre-existing target ligand file path

## Output Structure

```
base_dir/
├── ligands/              # Stage 1 output
│   ├── from_pdb/
│   ├── from_pubchem/
│   └── from_smiles/
├── target_std/           # Stage 2 output (standardized)
├── target_blurred/       # Stage 2 output (backbone-only)
├── configs/              # Stage 3 output
├── runs/                 # Stage 4+ output
├── status/               # Status tracking files
│   └── <complex_id>/
│       ├── 1_save_ligands.success
│       ├── 2_prep_target.success
│       └── ...
├── stage1_output.csv
├── stage2_output.csv
├── stage3_output.csv
└── pipeline_progress.csv # Final progress report
```

## Status Tracking

Each complex has a status directory with files indicating success/failure:
- `<stage>.success` - Stage completed successfully
- `<stage>.failed` - Stage failed (contains error message)
- `<stage>.skip` - Stage skipped due to previous failure

## Progress Report

The final `pipeline_progress.csv` contains:
- Complex ID and metadata
- Status for each stage (success/failed/skip/pending)
- Error messages for failed stages
- Overall pipeline statistics

## Example

```bash
# Prepare training data
python pipeline.py \
  --csv /scratch/data/train_set_sampled_20targets_100offtargets.csv \
  --base_dir /scratch/data/train_example \
  --template config.yaml

# Check progress
python pipeline.py \
  --csv /scratch/data/train_set_sampled_20targets_100offtargets.csv \
  --base_dir /scratch/data/train_example \
  --template config.yaml \
  --report-only
```

## Monitoring

Monitor stage completion:
```bash
# Count success/failure for each stage
find base_dir/status -name "1_save_ligands.success" | wc -l
find base_dir/status -name "1_save_ligands.failed" | wc -l

# View failed cases
find base_dir/status -name "*.failed" -exec cat {} \;
```

## Requirements

- Python 3.7+
- BioPython
- RDKit
- PyYAML
- pandas
- requests
- AutoDock Vina (for stages 5+)
- meeko (for PDBQT conversion)
- scipy (for clustering in stage 6)
- numpy

Install with:
```bash
pip install biopython rdkit pyyaml pandas requests meeko scipy numpy
```

Install AutoDock Vina from: https://github.com/ccsb-scripps/AutoDock-Vina
