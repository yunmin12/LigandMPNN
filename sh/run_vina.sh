BASE_DIR=/scratch/yunmin/data/db/PYR1/inputs/vina
OUT_DIR_PREFIX="$1"
OUT_DIR="${BASE_DIR}/${OUT_DIR_PREFIX}"

set -euo pipefail

# seed 1
vina --receptor ${BASE_DIR}/7MWN_receptor_prep.pdbqt \
  --ligand ${BASE_DIR}/A8S_ligand_aligned.pdbqt \
  --config ${BASE_DIR}/7MWN_receptor_prep.box.txt \
  --exhaustiveness 32 \
  --num_modes 40 \
  --energy_range 6 \
  --seed 1 \
  --out ${OUT_DIR}/7MWN_A8S_vina_out_seed1.pdbqt

# seed 2
vina --receptor ${BASE_DIR}/7MWN_receptor_prep.pdbqt \
  --ligand ${BASE_DIR}/A8S_ligand_aligned.pdbqt \
  --config ${BASE_DIR}/7MWN_receptor_prep.box.txt \
  --exhaustiveness 32 \
  --num_modes 40 \
  --energy_range 6 \
  --seed 2 \
  --out ${OUT_DIR}/7MWN_A8S_vina_out_seed2.pdbqt

# seed 3
vina --receptor ${BASE_DIR}/7MWN_receptor_prep.pdbqt \
  --ligand ${BASE_DIR}/A8S_ligand_aligned.pdbqt \
  --config ${BASE_DIR}/7MWN_receptor_prep.box.txt \
  --exhaustiveness 32 \
  --num_modes 40 \
  --energy_range 6 \
  --seed 3 \
  --out ${OUT_DIR}/7MWN_A8S_vina_out_seed3.pdbqt

# Merge all the outputs
echo "[INFO] Start merging output files..."
cat ${OUT_DIR}/7MWN_A8S_vina_out_seed*.pdbqt > ${OUT_DIR}/7MWN_A8S_vina_out_merged.pdbqt
echo "[INFO] Merged all the outputs!"

# Filtering top pose
python /home/yunmin/proj/LigandMPNN/dataprep/vina_filter.py \
  --ref_ligand ${BASE_DIR}/A8S_ligand_aligned.pdbqt \
  --protein ${BASE_DIR}/7MWN_receptor_prep.pdbqt \
  --vina_out ${OUT_DIR}/7MWN_A8S_vina_out_merged.pdbqt \
  --max_com_dist 4.0 \
  --energy_window 3.0 \
  --clash_cutoff 2.0 \
  --max_poses 10
