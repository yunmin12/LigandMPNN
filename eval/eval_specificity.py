#!/usr/bin/env python3
import json, math, argparse, numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)

def js(p, q, eps=1e-12):
    m = 0.5*(p+q)
    return 0.5*kl(p,m,eps)+0.5*kl(q,m,eps)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def site_mask_from_pdb(pdb_path, cutoff=6.0):
    # quick parser: protein CA, ligand HETATM (non water), no deps
    protein = []; ligand=[]
    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")): continue
            name=line[12:16].strip()
            resn=line[17:20].strip()
            chain=line[21].strip()
            resi=int(line[22:26])
            x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
            is_water = (resn in ("HOH","WAT"))
            if line.startswith("ATOM") and name=="CA":
                protein.append(((chain, resi), (x,y,z)))
            elif line.startswith("HETATM") and (not is_water):
                ligand.append((x,y,z))
    if not protein or not ligand:
        return None
    L = len(protein)
    lig = np.array(ligand)            # [Na,3]
    ca  = np.array([p[1] for p in protein])  # [L,3]
    # min distance of residue to any ligand atom
    d = np.sqrt(((ca[:,None,:]-lig[None,:,:])**2).sum(-1)).min(axis=1)
    mask = (d <= cutoff).astype(np.bool_)
    # order by residue index: run.py의 residue 순서와 동일하다고 가정
    return mask  # [L]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats_json", required=True)        # contrastive=on 결과
    ap.add_argument("--baseline_json", default="")        # contrastive=off 결과(선택)
    ap.add_argument("--target_pdb", required=True)        # 포켓 정의용
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    J = load_json(args.stats_json)
    L21 = J.get("contrastive_per_ligand_logits_step", None)   # [B,L,21]
    F21 = J.get("contrastive_fused_logits_step", None)        # [L,21]
    S   = J["generated_sequences"][0] if "generated_sequences" in J else None  # [L] (int)
    if L21 is None or F21 is None:
        raise RuntimeError("Missing per-ligand or fused logits in stats JSON.")
    L21 = np.asarray(L21, dtype=np.float64)  # [B,L,21]
    F21 = np.asarray(F21, dtype=np.float64)  # [L,21]
    B, L, A = L21.shape
    assert A==21

    X_idx = 20
    # mask out X channel if needed
    L21[..., X_idx] = -1e9
    F21[..., X_idx] = -1e9

    t = L21[0]            # [L,21]
    offs = L21[1:]        # [B-1,L,21]
    if offs.size == 0:
        raise RuntimeError("No off-target logits present (B must be >=2).")
    off_mean = offs.mean(axis=0)   # [L,21]

    # 1) Global AA-wise margin
    margin = (t - off_mean).mean()    # scalar

    # 2) JS divergence between distributions (per residue, avg)
    p_t = softmax(t/args.temperature, axis=-1)           # [L,21]
    p_off_mean = softmax(off_mean/args.temperature, -1)  # [L,21]
    js_per_res = js(p_t, p_off_mean)                     # [L]
    js_global = js_per_res.mean()

    # 3) Binding-site change rate (argmax change t vs fused, at pocket)
    arg_t = np.argmax(t, axis=-1)        # [L]
    arg_f = np.argmax(F21, axis=-1)      # [L]
    site_mask = site_mask_from_pdb(args.target_pdb, args.cutoff)
    if site_mask is None:
        site_change_rate = float("nan")
        site_L = 0
    else:
        ch = (arg_t != arg_f) & site_mask
        site_L = int(site_mask.sum())
        site_change_rate = ch.sum() / max(1, site_L)

    # 4) Off-target score drop for designed seq (avg per-residue log-prob diff)
    #    Use fused-designed arg_f as the designed AA (or S if provided).
    designed = arg_f if S is None else np.asarray(S, dtype=np.int64)
    # log-prob under target/off_mean (same temperature)
    t_lp = (t/args.temperature) - np.log(np.sum(np.exp(t/args.temperature), axis=-1, keepdims=True))  # [L,21]
    o_lp = (off_mean/args.temperature) - np.log(np.sum(np.exp(off_mean/args.temperature), axis=-1, keepdims=True))
    pick = np.arange(L)
    t_ll = t_lp[pick, designed].mean()
    o_ll = o_lp[pick, designed].mean()
    off_drop = t_ll - o_ll  # positive means worse on off-target (good for specificity)

    # print summary
    print(f"[L={L} A=21 B={B}]")
    print(f"AA-wise margin (t - off_mean): {margin:.4f}")
    print(f"JS(target || off_mean): {js_global:.4f}")
    if site_mask is not None:
        print(f"Binding-site argmax-change rate (<= {args.cutoff}Å): {site_change_rate:.4f} (site L={site_L})")
    else:
        print(f"Binding-site argmax-change rate: NA (ligand/protein not found)")
    print(f"Off-target score drop for designed seq (target_ll - off_ll): {off_drop:.4f}")

    # optional: dump detailed arrays if needed
    # np.savez("eval_debug.npz", t=t, off_mean=off_mean, fused=F21, site_mask=site_mask, designed=designed)

if __name__ == "__main__":
    main()
