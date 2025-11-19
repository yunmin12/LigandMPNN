protein=$1
ligand=$2
old_resname=$3
old_chain=$4
old_resid=$5
python /home/yunmin/proj/LigandMPNN/scripts/substitute.py \
--complex_pdb /scratch/yunmin/data/db/bench/${protein}.pdb \
--new_ligand /scratch/yunmin/data/db/bench/${ligand}.mol2 \
--old_resname ${old_resname} \
--old_chain ${old_chain} \
--old_resid ${old_resid} \
--out_pdb /scratch/yunmin/data/rfdaa/${protein}
