set -euo pipefail
shopt -s nullglob

# Usage Example
#BASE_DIR=/scratch/yunmin/data/db/v2/P00519_T315I/split_off_large
#CSV=/scratch/yunmin/data/db/v2/ensemble_off_summary_large.csv
BASE_DIR="$2"
CSV="$3"

LABEL=${1:? "label value required"}

mapfile -t ZERO_BASED_INDICES < <(
  awk -v label="$LABEL" -v FPAT='([^,]*)|("([^"]*)")' '
    NR==1{
      for(i=1;i<=NF;i++){
        h=$i; gsub(/^ *"|" *$|[[:space:]]+/,"",h)
        if(h=="label") lcol=i
        if(h=="ranked_model_indices") rcol=i
      }
      next
    }
    {
      lb=$lcol; gsub(/^ *"|" *$|[[:space:]]+/,"",lb)
      if(lb==label){
        v=$rcol
        gsub(/^ *"|" *$|[[:space:]]+/,"",v)
        n=split(v,a,/ *, */)
        for(i=1;i<=n;i++){
          if(a[i] ~ /^[0-9]+$/){
            print a[i]
          }
        }
        exit
      }
    }
  ' "$CSV"
)

declare -A KEEP=()
for n in "${ZERO_BASED_INDICES[@]}"; do
  printf -v key "%02d" "$n"
  KEEP["$key"]=1
done

kept=0; deleted=0
for f in ${BASE_DIR}/"${LABEL}"_*.pdb; do
  base=${f##*/}
  num=${base%.pdb}; num=${num##*_}
  if [[ -n "${KEEP[$num]:-}" ]]; then
    echo "Keeping  $f"
    ((kept+=1))
  else
    echo "Deleting $f"
    rm -f -- "$f"
    ((deleted+=1))
  fi
done

printf "%s\n" "${!KEEP[@]}" | sort -V | while read -r k; do
    [[ -n "$k" && "${KEEP[$k]:-}" == 1 ]] && printf "%s," "$k"
done | sed 's/,$/\n/'
echo "Done. kept=$kept, deleted=$deleted"
