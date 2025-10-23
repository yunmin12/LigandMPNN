set -euo pipefail
shopt -s nullglob

base_dir="/home/yunmin/proj/data/db/v2/"
mode="${1:?mode required: tar|off}"

case "$mode" in
  tar) placer_dirname="placer_tar"; split_dirname="split_tar" ;;
  off) placer_dirname="placer_off"; split_dirname="split_off" ;;
  *) echo "mode must be tar or off" >&2; exit 1 ;;
esac

for placer_dir in "$base_dir"/*/"$placer_dirname"; do
  [ -d "$placer_dir" ] || continue
  parent_dir="$(dirname "$placer_dir")"
  split_dir="$parent_dir/$split_dirname"
  mkdir -p "$split_dir"

  for multi in "$placer_dir"/*/*/*_model.pdb; do
    [ -f "$multi" ] || continue
    bn="$(basename "$multi" .pdb)"
    echo $multi
    awk -v out_prefix="$split_dir/${bn}_" '
      /^MODEL[[:space:]]/ { n++; close(out); fn=sprintf("%s%02d.pdb", out_prefix, n-1); out=fn; next }
      /^ENDMDL/ { close(out); next }
      { if (out=="") { n=1; fn=sprintf("%s%02d.pdb", out_prefix, 0); out=fn } print >> out }
      END { if (out!="") close(out) }
    ' "$multi"
  done
done

echo "Split done."
