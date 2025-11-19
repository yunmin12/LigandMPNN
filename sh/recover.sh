protein="$1"
cd $protein
cd outputs/

for i in {0..4}; do
	  in="sub_test_${i}_model.pdb"
	  split_dir="split_${i}"
	  mkdir -p "$split_dir"
	
	  awk -v d="$split_dir" '
	    BEGIN { n=0; out="" }
	    /^MODEL[[:space:]]/ {
	      if (out) { close(out) }
	      n++
	      out = sprintf("%s/ens_%04d.pdb", d, n)
	      next
	    }
	    
	    /^ENDMDL/ {
	      if (out) { close(out); out="" }
	      next
	    }

	    out { print > out }
	    END { if (out) close(out) }
	  ' "$in"

  	printf "%s: %d\n" "$split_dir" "$(ls -1 "$split_dir"/*.pdb 2>/dev/null | wc -l)"

	python /home/yunmin/proj/LigandMPNN/scripts/recover.py \
		--full_pdb /scratch/yunmin/data/placer/${protein}/sub_test_${i}.pdb \
		--split_dir /scratch/yunmin/data/placer/${protein}/outputs/split_${i} \
		--out_dir /scratch/yunmin/data/input/0911/${protein}/split_${i};
done

