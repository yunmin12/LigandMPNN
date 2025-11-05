import argparse
import json
import os.path
import random
import sys

import numpy as np
import torch

from data_utils import (
    element_dict_rev,
    alphabet,
    restype_int_to_str,
    featurize,
    parse_PDB,
)
from model_utils import ProteinMPNN


def main(args) -> None:
    """
    Inference function
    """
    if args.seed:
        seed = args.seed
    else:
        seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    folder_for_outputs = args.out_folder
    base_folder = folder_for_outputs
    if base_folder[-1] != "/":
        base_folder = base_folder + "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder, exist_ok=True)
    if args.model_type == "protein_mpnn":
        checkpoint_path = args.checkpoint_protein_mpnn
    elif args.model_type == "ligand_mpnn":
        checkpoint_path = args.checkpoint_ligand_mpnn
    elif args.model_type == "per_residue_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_per_residue_label_membrane_mpnn
    elif args.model_type == "global_label_membrane_mpnn":
        checkpoint_path = args.checkpoint_global_label_membrane_mpnn
    elif args.model_type == "soluble_mpnn":
        checkpoint_path = args.checkpoint_soluble_mpnn
    else:
        print("Choose one of the available models")
        sys.exit()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if args.model_type == "ligand_mpnn":
        atom_context_num = checkpoint["atom_context_num"]
        ligand_mpnn_use_side_chain_context = args.ligand_mpnn_use_side_chain_context
        k_neighbors = checkpoint["num_edges"]
    else:
        atom_context_num = 1
        ligand_mpnn_use_side_chain_context = 0
        k_neighbors = checkpoint["num_edges"]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=args.model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if args.pdb_path_multi:
        with open(args.pdb_path_multi, "r") as fh:
            pdb_paths = list(json.load(fh))
    else:
        pdb_paths = [args.pdb_path]

    if args.fixed_residues_multi:
        with open(args.fixed_residues_multi, "r") as fh:
            fixed_residues_multi = json.load(fh)
    else:
        fixed_residues = [item for item in args.fixed_residues.split()]
        fixed_residues_multi = {}
        for pdb in pdb_paths:
            fixed_residues_multi[pdb] = fixed_residues

    if args.redesigned_residues_multi:
        with open(args.redesigned_residues_multi, "r") as fh:
            redesigned_residues_multi = json.load(fh)
    else:
        redesigned_residues = [item for item in args.redesigned_residues.split()]
        redesigned_residues_multi = {}
        for pdb in pdb_paths:
            redesigned_residues_multi[pdb] = redesigned_residues

    def _normalize_off_target_list(raw_value):
        if raw_value is None:
            return []
        if isinstance(raw_value, list):
            items = []
            for entry in raw_value:
                items.extend(_normalize_off_target_list(entry))
            return items
        if isinstance(raw_value, str):
            if raw_value.strip() == "":
                return []
            return [item.strip() for item in raw_value.split(",") if item.strip()]
        raise ValueError(f"Unsupported off-target entry type: {type(raw_value)}")

    def _compute_auto_pocket_mask(feature_dict, cutoff_radius):
        if "Y" not in feature_dict or "Y_m" not in feature_dict:
            return None
        X = feature_dict.get("X")
        mask = feature_dict.get("mask")
        if X is None or mask is None:
            return None
        X = X[0]
        mask = mask[0]
        Y = feature_dict["Y"][0]
        Y_m = feature_dict["Y_m"][0]
        if Y.numel() == 0 or torch.sum(Y_m) == 0:
            return None
        N = X[:, 0, :]
        CA = X[:, 1, :]
        C = X[:, 2, :]
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        diff = Y - CB[:, None, :]
        distances = torch.linalg.norm(diff, dim=-1)
        distances = distances.masked_fill(Y_m == 0, float("inf"))
        min_distances = torch.min(distances, dim=-1).values
        pocket_mask = ((min_distances <= cutoff_radius) & torch.isfinite(min_distances)).float() * mask
        return pocket_mask.unsqueeze(0)

    global_off_targets = _normalize_off_target_list(args.off_target_pdb_path)
    legacy_global_off_targets = _normalize_off_target_list(args.offtarget_pdb_path)
    combined_global_off_targets = list(
        dict.fromkeys(global_off_targets + legacy_global_off_targets)
    )
    off_target_map = {pdb: list(combined_global_off_targets) for pdb in pdb_paths}
    if args.off_target_pdb_path_multi:
        with open(args.off_target_pdb_path_multi, "r") as fh:
            off_target_multi = json.load(fh)
        for key, value in off_target_multi.items():
            combined = off_target_map.get(key, list(global_off_targets)) + _normalize_off_target_list(value)
            off_target_map[key] = list(dict.fromkeys(combined))

    # loop over PDB paths
    for pdb in pdb_paths:
        if args.verbose:
            print("Designing protein from this path:", pdb)
        fixed_residues = fixed_residues_multi[pdb]
        redesigned_residues = redesigned_residues_multi[pdb]
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            pdb,
            device=device,
            chains=args.parse_these_chains_only,
            parse_all_atoms=args.ligand_mpnn_use_side_chain_context,
            parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy
        )
        # make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(
            zip(list(range(len(encoded_residues))), encoded_residues)
        )

        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=device,
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues],
            device=device,
        )

        # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
        if args.transmembrane_buried:
            buried_residues = [item for item in args.transmembrane_buried.split()]
            buried_positions = torch.tensor(
                [int(item in buried_residues) for item in encoded_residues],
                device=device,
            )
        else:
            buried_positions = torch.zeros_like(fixed_positions)

        if args.transmembrane_interface:
            interface_residues = [item for item in args.transmembrane_interface.split()]
            interface_positions = torch.tensor(
                [int(item in interface_residues) for item in encoded_residues],
                device=device,
            )
        else:
            interface_positions = torch.zeros_like(fixed_positions)
        protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        if args.model_type == "global_label_membrane_mpnn":
            protein_dict["membrane_per_residue_labels"] = (
                args.global_transmembrane_label + 0 * fixed_positions
            )
        if type(args.chains_to_design) == str:
            chains_to_design_list = args.chains_to_design.split(",")
        else:
            chains_to_design_list = protein_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in protein_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=device,
        )

        # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        if args.verbose:
            PDB_residues_to_be_redesigned = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 1
            ]
            PDB_residues_to_be_fixed = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 0
            ]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)

        # specify which residues are linked
        if args.symmetry_residues:
            symmetry_residues_list_of_lists = [
                x.split(",") for x in args.symmetry_residues.split("|")
            ]
            remapped_symmetry_residues = []
            for t_list in symmetry_residues_list_of_lists:
                tmp_list = []
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list)
        else:
            remapped_symmetry_residues = [[]]

        if args.homo_oligomer:
            if args.verbose:
                print("Designing HOMO-OLIGOMER")
            chain_letters_set = list(set(chain_letters_list))
            reference_chain = chain_letters_set[0]
            lc = len(reference_chain)
            residue_indices = [
                item[lc:] for item in encoded_residues if item[:lc] == reference_chain
            ]
            remapped_symmetry_residues = []
            for res in residue_indices:
                tmp_list = []
                tmp_w_list = []
                for chain in chain_letters_set:
                    name = chain + res
                    tmp_list.append(encoded_residue_dict[name])
                    tmp_w_list.append(1 / len(chain_letters_set))
                remapped_symmetry_residues.append(tmp_list)

        # set other atom bfactors to 0.0
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors * 0.0)

        # adjust input PDB name by dropping .pdb if it does exist
        name = pdb[pdb.rfind("/") + 1 :]
        if name[-4:] == ".pdb":
            name = name[:-4]

        with torch.no_grad():
            # run featurize to remap R_idx and add batch dimension
            if args.verbose:
                if "Y" in list(protein_dict):
                    atom_coords = protein_dict["Y"].cpu().numpy()
                    atom_types = list(protein_dict["Y_t"].cpu().numpy())
                    atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                    number_of_atoms_parsed = np.sum(atom_mask)
                else:
                    print("No ligand atoms parsed")
                    number_of_atoms_parsed = 0
                    atom_types = ""
                    atom_coords = []
                if number_of_atoms_parsed == 0:
                    print("No ligand atoms parsed")
                elif args.model_type == "ligand_mpnn":
                    print(
                        f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                    )
                    for i, atom_type in enumerate(atom_types):
                        print(
                            f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                        )
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=args.ligand_mpnn_cutoff_for_score,
                use_atom_context=args.ligand_mpnn_use_atom_context,
                number_of_ligand_atoms=atom_context_num,
                model_type=args.model_type,
            )
            feature_dict["batch_size"] = args.batch_size
            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
            # add additional keys to the feature dictionary
            feature_dict["symmetry_residues"] = remapped_symmetry_residues

            current_off_targets = off_target_map.get(pdb, list(combined_global_off_targets))
            off_feature_dicts = []
            for off_path in current_off_targets:
                off_protein_dict, _, _, off_icodes, _ = parse_PDB(
                    off_path,
                    device=device,
                    chains=args.parse_these_chains_only,
                    parse_all_atoms=args.ligand_mpnn_use_side_chain_context,
                    parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
                )
                off_encoded_residues = []
                off_r_idx_list = list(off_protein_dict["R_idx"].cpu().numpy())
                off_chain_letters = list(off_protein_dict["chain_letters"])
                for idx_tmp, r_idx_item in enumerate(off_r_idx_list):
                    off_encoded_residues.append(
                        str(off_chain_letters[idx_tmp]) + str(r_idx_item) + off_icodes[idx_tmp]
                    )
                if off_encoded_residues != encoded_residues:
                    raise ValueError(
                        f"Residue mapping mismatch between target {pdb} and off-target {off_path}."
                    )
                off_protein_dict["chain_mask"] = protein_dict["chain_mask"]
                if "membrane_per_residue_labels" in protein_dict:
                    off_protein_dict["membrane_per_residue_labels"] = protein_dict[
                        "membrane_per_residue_labels"
                    ]
                off_feature_dict = featurize(
                    off_protein_dict,
                    cutoff_for_score=args.ligand_mpnn_cutoff_for_score,
                    use_atom_context=args.ligand_mpnn_use_atom_context,
                    number_of_ligand_atoms=atom_context_num,
                    model_type=args.model_type,
                )
                off_feature_dict["batch_size"] = args.batch_size
                off_feature_dict["symmetry_residues"] = remapped_symmetry_residues
                off_feature_dicts.append(off_feature_dict)

            feature_dict["off_target_features"] = off_feature_dicts
            mask_components = []
            if args.auto_pocket:
                auto_mask = _compute_auto_pocket_mask(feature_dict, args.auto_pocket_cutoff)
                if auto_mask is not None:
                    mask_components.append(auto_mask)
                else:
                    if args.verbose:
                        print("[Warning] auto pocket requested but ligand context absent; falling back to full mask.")
            if getattr(args, "negative_residues", ""):
                idxs = []
                for tok in str(args.negative_residues).strip().split():
                    try:
                        idx_val = int(tok)
                        if 0 <= idx_val < L:
                            idxs.append(idx_val)
                    except ValueError:
                        continue
                if idxs:
                    manual_mask = torch.zeros((1, L), dtype=torch.float32, device=device)
                    manual_mask[:, idxs] = 1.0
                    mask_components.append(manual_mask)
            if mask_components:
                penalty_mask_tensor = torch.ones((1, L), dtype=torch.float32, device=device)
                for mask_component in mask_components:
                    penalty_mask_tensor = penalty_mask_tensor * mask_component
                feature_dict["off_target_residue_mask"] = penalty_mask_tensor

            target_weight = float(args.target_logit_weight)
            off_weight = float(args.off_target_logit_weight)
            if getattr(args, "negative_enable", 0):
                off_weight = float(args.negative_weight)
            feature_dict["target_weight"] = target_weight
            feature_dict["off_target_weight"] = off_weight

            logits_list = []
            probs_list = []
            log_probs_list = []
            decoding_order_list = []
            for _ in range(args.number_of_batches):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=device,
                )
                for off_fd in feature_dict["off_target_features"]:
                    off_fd["randn"] = feature_dict["randn"]
                if args.autoregressive_score:
                    score_dict = model.score(feature_dict, use_sequence=args.use_sequence)
                elif args.single_aa_score:
                    score_dict = model.single_aa_score(feature_dict, use_sequence=args.use_sequence)
                else:
                    print("Set either autoregressive_score or single_aa_score to True")
                    sys.exit()
                logits_list.append(score_dict["logits"])
                log_probs_list.append(score_dict["log_probs"])
                probs_list.append(torch.exp(score_dict["log_probs"]))
                decoding_order_list.append(score_dict["decoding_order"])
            log_probs_stack = torch.cat(log_probs_list, 0)
            logits_stack = torch.cat(logits_list, 0)
            probs_stack = torch.cat(probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)

            output_stats_path = base_folder + name + args.file_ending + ".pt"
            out_dict = {}
            out_dict["logits"] = logits_stack.cpu().numpy()
            out_dict["probs"] = probs_stack.cpu().numpy()
            out_dict["log_probs"] = log_probs_stack.cpu().numpy()
            out_dict["decoding_order"] = decoding_order_stack.cpu().numpy()
            out_dict["native_sequence"] = feature_dict["S"][0].cpu().numpy()
            out_dict["mask"] = feature_dict["mask"][0].cpu().numpy()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu().numpy() #this affects decoding order
            out_dict["seed"] = seed
            out_dict["alphabet"] = alphabet
            out_dict["residue_names"] = encoded_residue_dict_rev

            mean_probs = np.mean(out_dict["probs"], 0)
            std_probs = np.std(out_dict["probs"], 0)
            sequence = [restype_int_to_str[AA] for AA in out_dict["native_sequence"]]
            mean_dict = {}
            std_dict = {}
            for residue in range(L):
                mean_dict_ = dict(zip(alphabet, mean_probs[residue]))
                mean_dict[encoded_residue_dict_rev[residue]] = mean_dict_
                std_dict_ = dict(zip(alphabet, std_probs[residue]))
                std_dict[encoded_residue_dict_rev[residue]] = std_dict_

            out_dict["sequence"] = sequence
            out_dict["mean_of_probs"] = mean_dict
            out_dict["std_of_probs"] = std_dict
            torch.save(out_dict, output_stats_path)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument(
        "--model_type",
        type=str,
        default="protein_mpnn",
        help="Choose your model: protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn, global_label_membrane_mpnn, soluble_mpnn",
    )
    # protein_mpnn - original ProteinMPNN trained on the whole PDB exluding non-protein atoms
    # ligand_mpnn - atomic context aware model trained with small molecules, nucleotides, metals etc on the whole PDB
    # per_residue_label_membrane_mpnn - ProteinMPNN model trained with addition label per residue specifying if that residue is buried or exposed
    # global_label_membrane_mpnn - ProteinMPNN model trained with global label per PDB id to specify if protein is transmembrane
    # soluble_mpnn - ProteinMPNN trained only on soluble PDB ids
    argparser.add_argument(
        "--checkpoint_protein_mpnn",
        type=str,
        default="/home/yunmin/proj/LigandMPNN/model_params/proteinmpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_ligand_mpnn",
        type=str,
        default="/home/yunmin/proj/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_per_residue_label_membrane_mpnn",
        type=str,
        default="/home/yunmin/proj/LigandMPNN/model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_global_label_membrane_mpnn",
        type=str,
        default="/home/yunmin/proj/LigandMPNN/model_params/global_label_membrane_mpnn_v_48_020.pt",
        help="Path to model weights.",
    )
    argparser.add_argument(
        "--checkpoint_soluble_mpnn",
        type=str,
        default="/home/yunmin/proj/LigandMPNN/model_params/solublempnn_v_48_020.pt",
        help="Path to model weights.",
    )

    argparser.add_argument("--verbose", type=int, default=1, help="Print stuff")

    argparser.add_argument(
        "--pdb_path", type=str, default="", help="Path to the input PDB."
    )
    argparser.add_argument(
        "--pdb_path_multi",
        type=str,
        default="",
        help="Path to json listing PDB paths. {'/path/to/pdb': ''} - only keys will be used.",
    )

    argparser.add_argument(
        "--off_target_pdb_path",
        type=str,
        default="",
        help="Comma-separated list of off-target PDB paths to include during scoring.",
    )
    argparser.add_argument(
        "--off_target_pdb_path_multi",
        type=str,
        default="",
        help="Path to JSON mapping each target PDB path to a list of off-target structures.",
    )
    argparser.add_argument(
        "--target_logit_weight",
        type=float,
        default=1.0,
        help="Weight applied to target logits before combining with off-target predictions.",
    )
    argparser.add_argument(
        "--off_target_logit_weight",
        type=float,
        default=1.0,
        help="Weight applied to the mean off-target logits before subtraction.",
    )
    argparser.add_argument(
        "--auto_pocket",
        type=int,
        default=0,
        help="If set to 1, restrict off-target penalties to residues within --auto_pocket_cutoff Angstroms of the ligand.",
    )
    argparser.add_argument(
        "--auto_pocket_cutoff",
        type=float,
        default=8.0,
        help="Distance cutoff (Angstrom) for defining the ligand pocket when --auto_pocket is enabled.",
    )

    argparser.add_argument(
        "--offtarget_pdb_path",
        action="append",
        default=None,
        help="Legacy repeatable flag for off-target PDBs; kept for backward compatibility.",
    )
    argparser.add_argument(
        "--negative_enable",
        type=int,
        default=0,
        help="Compatibility toggle for legacy contrastive scoring.",
    )
    argparser.add_argument(
        "--negative_weight",
        type=float,
        default=1.0,
        help="Penalty weight when --negative_enable is set.",
    )
    argparser.add_argument(
        "--negative_residues",
        type=str,
        default="",
        help="Space-separated zero-based residue indices where off-target penalties apply.",
    )

    argparser.add_argument(
        "--fixed_residues",
        type=str,
        default="",
        help="Provide fixed residues, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--fixed_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of fixed residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--redesigned_residues",
        type=str,
        default="",
        help="Provide to be redesigned residues, everything else will be fixed, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--redesigned_residues_multi",
        type=str,
        default="",
        help="Path to json mapping of redesigned residues for each pdb i.e., {'/path/to/pdb': 'A12 A13 A14 B2 B25'}",
    )

    argparser.add_argument(
        "--symmetry_residues",
        type=str,
        default="",
        help="Add list of lists for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'",
    )
    
    argparser.add_argument(
        "--homo_oligomer",
        type=int,
        default=0,
        help="Setting this to 1 will automatically set --symmetry_residues and --symmetry_weights to do homooligomer design with equal weighting.",
    )

    argparser.add_argument(
        "--out_folder",
        type=str,
        help="Path to a folder to output scores, e.g. /home/out/",
    )
    argparser.add_argument(
        "--file_ending", type=str, default="", help="adding_string_to_the_end"
    )
    argparser.add_argument(
        "--zero_indexed",
        type=str,
        default=0,
        help="1 - to start output PDB numbering with 0",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set seed for torch, numpy, and python random.",
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of sequence to generate per one pass.",
    )
    argparser.add_argument(
        "--number_of_batches",
        type=int,
        default=1,
        help="Number of times to design sequence using a chosen batch size.",
    )

    argparser.add_argument(
        "--ligand_mpnn_use_atom_context",
        type=int,
        default=1,
        help="1 - use atom context, 0 - do not use atom context.",
    )

    argparser.add_argument(
        "--ligand_mpnn_use_side_chain_context",
        type=int,
        default=0,
        help="Flag to use side chain atoms as ligand context for the fixed residues",
    )

    argparser.add_argument(
        "--ligand_mpnn_cutoff_for_score",
        type=float,
        default=8.0,
        help="Cutoff in angstroms between protein and context atoms to select residues for reporting score.",
    )

    argparser.add_argument(
        "--chains_to_design",
        type=str,
        default=None,
        help="Specify which chains to redesign, all others will be kept fixed.",
    )

    argparser.add_argument(
        "--parse_these_chains_only",
        type=str,
        default="",
        help="Provide chains letters for parsing backbones, 'ABCF'",
    )

    argparser.add_argument(
        "--transmembrane_buried",
        type=str,
        default="",
        help="Provide buried residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )
    argparser.add_argument(
        "--transmembrane_interface",
        type=str,
        default="",
        help="Provide interface residues when using checkpoint_per_residue_label_membrane_mpnn model, A12 A13 A14 B2 B25",
    )

    argparser.add_argument(
        "--global_transmembrane_label",
        type=int,
        default=0,
        help="Provide global label for global_label_membrane_mpnn model. 1 - transmembrane, 0 - soluble",
    )

    argparser.add_argument(
        "--parse_atoms_with_zero_occupancy",
        type=int,
        default=0,
        help="To parse atoms with zero occupancy in the PDB input files. 0 - do not parse, 1 - parse atoms with zero occupancy",
    )

    argparser.add_argument(
        "--use_sequence",
        type=int,
        default=1,
        help="1 - get scores using amino acid sequence info; 0 - get scores using backbone info only",
    )

    argparser.add_argument(
        "--autoregressive_score",
        type=int,
        default=0,
        help="1 - run autoregressive scoring function; p(AA_1|backbone); p(AA_2|backbone, AA_1) etc, 0 - False",
    )

    argparser.add_argument(
        "--single_aa_score",
        type=int,
        default=1,
        help="1 - run single amino acid scoring function; p(AA_i|backbone, AA_{all except ith one}), 0 - False",
    )

    args = argparser.parse_args()
    main(args)
