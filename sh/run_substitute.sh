protein=$1
ligand=$2
old_resname=$3
old_chain=$4
old_resid=$5
python /home/yunmin/proj/LigandMPNN/scripts/substitute.py \
--complex_pdb ${protein} \
--new_ligand ${ligand} \
--old_resname ${old_resname} \
--old_chain ${old_chain} \
--old_resid ${old_resid} \
--out_pdb /home/yunmin/proj/data/rfdaa/${protein}
