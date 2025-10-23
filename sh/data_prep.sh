$sh_dir="/home/yunmin/proj/LigandMPNN/sh"

# 1. Clean target and off-target complex PDB
bash $sh_dir/clean_hetatm.sh tar
bash $sh_dir/clean_hetatm.sh off
echo "✅ [clean] done."

# 2. Run PLACER to get an ensemble PDB
sbatch $sh_dir/run_placer_v2.sh tar
echo "✅ [PLACER] target done."
sbatch $sh_dir/run_placer_v2.sh off
echo "✅ [PLACER] off-target done."

# 3. Split and recover PLACER output
bash $sh_dir/split.sh tar
bash $sh_dir/split.sh off
echo "✅ [split] done."

python /home/yunmin/proj/LigandMPNN/scripts/recover_v2.py \
    $base_dir \
    --mode tar
echo "✅ [recover] target done."

python /home/yunmin/proj/LigandMPNN/scripts/recover_v2.py \
    $base_dir \
    --mode off
echo "✅ [recover] off-target done."
