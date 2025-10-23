TARGET_CHAIN="Z"
base_dir="/home/yunmin/proj/data/db/v2"

target="$1"
pattern="standard"
[ "$target" = "off" ] && pattern="replaced"

for replaced_dir in "$base_dir"/*/"$pattern"; do
    [ -d "$replaced_dir" ] || continue
    parent_dir=$(dirname "$replaced_dir")
    
    if [ "$target" == "tar" ]; then
        cleaned_dir="$parent_dir/cleaned_tar"
    elif [ "$target" == "off" ]; then
        cleaned_dir="$parent_dir/cleaned_off"
    else
        echo "Unknown target type: $target" >&2
        exit 1
    fi
    mkdir -p "$cleaned_dir"

    for pdb in "$replaced_dir"/*.pdb; do
        [ -f "$pdb" ] || continue
        base="$(basename "$pdb" .pdb)"
        protein_only="${cleaned_dir}/${base}_protein.tmp"
        ligand_only="${cleaned_dir}/${base}_ligand.tmp"
        out="${cleaned_dir}/${base}.pdb"
        
        echo "[Processing] $pdb → $out"
        pdb_delhetatm "$pdb" > "$protein_only"
        pdb_selhetatm "$pdb" | pdb_selchain -${TARGET_CHAIN} > "$ligand_only"
        cat "$protein_only" "$ligand_only" | pdb_tidy | pdb_reatom > "$out"
        rm -f "$protein_only" "$ligand_only"
    done
done
