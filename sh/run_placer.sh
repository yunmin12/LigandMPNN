protein="$1"
ligand="$2"
cd ${protein}
for i in {0..4}; do
	placer --ifile sub_test_${i}.pdb \
		--odir outputs \
		--rerank prmsd \
		-n 50 \
		--corruption_centers B-427-A8S-O7 B-427-A8S-C8 B-427-A8S-C9 \
		--predict_ligand ${ligand};
done
