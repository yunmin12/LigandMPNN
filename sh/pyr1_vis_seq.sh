BASE_DIR="/scratch/yunmin/data/db/PYR1"

python /home/yunmin/proj/LigandMPNN/eval/compare_seq.py \
  --in_dirs $BASE_DIR/lmpnn_out_1128_auto_pocket/lmpnn_tar \
            $BASE_DIR/lmpnn_out_1128_auto_pocket/lmpnn_mod \
  --dir_labels "tar" "mod" \
  --reference_fasta $BASE_DIR/inputs/3QN1_wt.fa \
  --pocket_tsv $BASE_DIR/lmpnn_out_1204_auto_pocket/lmpnn_mod/auto_pocket_residues.tsv \
  --output_dir $BASE_DIR/plots_seq_1204/auto_pocket_2 \
