BASE_DIR="/home/yunmin/proj/data/db/v2"
OUT_DIR="$1"
PREFIX="$2"
KEY_MUT="$3"

shopt -s nullglob
for PREFIX_DIR in "${BASE_DIR}/${OUT_DIR}_v2"/*/*; do
  TGT_PREFIX=$( basename $PREFIX_DIR )
  for OFF_DIR in $PREFIX_DIR/*; do
    OFF_PREFIX=$( basename $OFF_DIR )
    if [ ${TGT_PREFIX%_*} != "${PREFIX%_*}" ]; then
      continue
    fi
    echo "📁 Directory: ${PREFIX%_*}, Target: ${PREFIX##*_}, Off-target: ${OFF_PREFIX##*_}"
    python /home/yunmin/proj/LigandMPNN/eval/vis_match.py \
      --orig_pt $BASE_DIR/${OUT_DIR}_v0/${PREFIX%_*_*}/${PREFIX}/scores \
      --v1_pt $BASE_DIR/${OUT_DIR}_v1/${PREFIX%_*_*}/${PREFIX}/${OFF_PREFIX##*_}/scores \
      --v2_pt $BASE_DIR/${OUT_DIR}_v2/${PREFIX%_*_*}/${PREFIX}/${OFF_PREFIX##*_}/scores \
      --key_mutations "$KEY_MUT" \
      --pdb_path $(ls $BASE_DIR/${OUT_DIR}_v0/${PREFIX%_*_*}/${PREFIX}/backbones/*.pdb | head -n 1 ) \
      --outdir $BASE_DIR/$OUT_DIR/${PREFIX}/${OFF_PREFIX##*_}
    echo "Plots saved at $BASE_DIR/$OUT_DIR/${PREFIX}/${OFF_PREFIX##*_}"
  done
done
shopt -u nullglob
echo "🙌 Done!"

