protein="$1"
case "$protein" in
  aldo_keto)
	python -u /home/yunmin/proj/LigandMPNN/scripts/score_vis.py \
	--results_dir ~/proj/data/bench/outputs/1001/scores \
	--out_dir ~/proj/data/bench/outputs/1001/scores/aldo_keto \
	--mutation 'A:298:C>A' --mutation 'A:219:W>Y' \
	--pdb_ids 2IPW,2IS7,2INZ,2INE
    ;;

  amine_oxidase)
	python -u /home/yunmin/proj/LigandMPNN/scripts/score_vis.py \
    --results_dir ~/proj/data/bench/outputs/1001/scores \
    --out_dir ~/proj/data/bench/outputs/1001/scores/amine_oxidase \
    --mutation 'A:199:F>I' \
    --pdb_ids 1S2Y,1OJA,2BK5
    ;;

  *)
    echo "Unknown protein: "$protein""
    exit 1
    ;;
esac
