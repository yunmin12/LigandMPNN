set -euo pipefail

BASE="/home/yunmin/proj/data/db/v2"
process_csv () {
  local CSV="$1"
  local TTYPE="$2"
  local OUTDIR="${BASE}/lmpnn_in_${TTYPE}"
  mkdir -p "$OUTDIR"

  echo "[INFO] Processing $CSV -> $OUTDIR"

  mapfile -t LINES < <(python - <<'PYCODE' "$CSV"
import sys, csv, json, re
csv_path = sys.argv[1]
pdb_pat = re.compile(r'^[0-9][A-Za-z0-9]{3}$')

with open(csv_path, newline='') as fh:
    rdr = csv.DictReader(fh)
    for r in rdr:
        label = (r.get("label") or "").strip()
        idx_raw = (r.get("ranked_model_indices") or "").strip()
        if not label or not idx_raw:
            continue

        try:
            idxs = json.loads(idx_raw)
            if isinstance(idxs, (int, float)): idxs = [int(idxs)]
            elif not isinstance(idxs, (list, tuple)):
                idxs = [int(x) for x in re.findall(r"\d+", idx_raw)]
        except Exception:
            idxs = [int(x) for x in re.findall(r"\d+", idx_raw)]
        if not idxs:
            continue

        toks = label.split('_')
        prt_mut_tokens = []
        for t in toks:
            if pdb_pat.match(t):
                break
            prt_mut_tokens.append(t)
        if len(prt_mut_tokens) < 2:
            prt_mut_tokens = toks[:2]
        prt_mut = "_".join(prt_mut_tokens)

        print(f"{label}\t{prt_mut}\t{' '.join(str(i) for i in idxs)}")
PYCODE
  )

  for line in "${LINES[@]}"; do
    IFS=$'\t' read -r LABEL PRT_MUT IDXLIST <<< "$line"
    for IDX in $IDXLIST; do
      PAD=$(printf "%02d" "$IDX")
      SRC="${BASE}/${PRT_MUT}/recover_${TTYPE}/${LABEL}_recovered_${PAD}.pdb"
      DEST="${OUTDIR}/${LABEL}_recover_${PAD}.pdb"
      if [[ -f "$SRC" ]]; then
        ln -sf "../${SRC}" "$DEST"
        echo "[OK] linked $SRC → $DEST"
      else
        echo "[WARN] not found: $SRC" >&2
      fi
    done
  done
}

[[ -f "${BASE}/ensemble_tar_summary.csv" ]] && process_csv "${BASE}/ensemble_tar_summary.csv" "tar"
[[ -f "${BASE}/ensemble_off_summary.csv" ]] && process_csv "${BASE}/ensemble_off_summary.csv" "off"

echo "✅ Done. Symlinks in lmpnn_in_tar/ and lmpnn_in_off/"

