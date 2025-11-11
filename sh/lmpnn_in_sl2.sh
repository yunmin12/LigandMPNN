set -euo pipefail

BASE_DIR="${BASE_DIR:-$HOME/proj/data/db/v2}"
OUT_TAR="${BASE_DIR}/lmpnn_in_tar"
OUT_OFF="${BASE_DIR}/lmpnn_in_off"

mkdir -p "$OUT_TAR" "$OUT_OFF"

parse_and_link () {
  local CSV="$1"
  local MODE="$2"  # tar/off
  local OUTROOT="${BASE_DIR}/lmpnn_in_${MODE}"

  echo "[INFO] CSV=$CSV MODE=$MODE OUTROOT=$OUTROOT"

  mapfile -t LINES < <(python - <<'PY' "$CSV"
import sys, csv, json, re
csv_path = sys.argv[1]
pdb_pat = re.compile(r'^[0-9][A-Za-z0-9]{3}$')

def get_prt_mut(label):
    toks = label.split('_')
    acc=[]
    for t in toks:
        if pdb_pat.match(t):
            break
        acc.append(t)
    if len(acc) < 2:
        acc = toks[:2]
    return "_".join(acc)

with open(csv_path, newline='') as fh:
    rdr = csv.DictReader(fh)
    for r in rdr:
        label = (r.get("label") or "").strip()
        idx_raw = (r.get("ranked_model_indices") or "").strip()
        if not label or not idx_raw:
            continue
        try:
            val = json.loads(idx_raw)
            if isinstance(val, (list, tuple)):
                idxs = [int(x) for x in val]
            else:
                idxs = [int(x) for x in re.findall(r"\d+", idx_raw)]
        except Exception:
            idxs = [int(x) for x in re.findall(r"\d+", idx_raw)]
        if not idxs:
            continue

        prt_mut = get_prt_mut(label)
        prefix  = label
        print(f"{label}\t{prt_mut}\t{prefix}\t"+ " ".join(map(str, idxs)))
PY
  )

  for line in "${LINES[@]}"; do
    IFS=$'\t' read -r LABEL PRT_MUT PREFIX IDXLIST <<< "$line"

    OUTDIR="${OUTROOT}/${PRT_MUT}/${PREFIX}"
    mkdir -p "$OUTDIR"

    for IDX in $IDXLIST; do
      PAD=$(printf "%02d" "$IDX")
      SRC="${BASE_DIR}/${PRT_MUT}/recover_${MODE}/${LABEL}_recovered_${PAD}.pdb"
      DEST="${OUTDIR}/${LABEL}_recovered_${PAD}.pdb"
      if [[ -f "$SRC" ]]; then
        ln -sf "../../../${PRT_MUT}/recover_${MODE}/${LABEL}_recovered_${PAD}.pdb" "$DEST"
        echo "[OK] ${MODE}: $DEST -> $SRC"
      else
        echo "[WARN] missing: $SRC" >&2
      fi
    done
  done
}

[[ -f "${BASE_DIR}/ensemble_tar_summary.csv" ]] && parse_and_link "${BASE_DIR}/ensemble_tar_summary.csv" "tar"
[[ -f "${BASE_DIR}/ensemble_off_summary.csv" ]] && parse_and_link "${BASE_DIR}/ensemble_off_summary.csv" "off"

echo "✅ Done. Structured symlinks at:"
echo "   - ${OUT_TAR}/{PRT_MUT}/{PREFIX}/"
echo "   - ${OUT_OFF}/{PRT_MUT}/{PREFIX}/"

