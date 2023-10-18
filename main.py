from era_utils import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=int, default=0, 
                    help="Path of Dataset. 0: DBP15k/zh_en/, 1: DBP15k/ja_en/, 2: DBP15k/fr_en/, 3: SRPRS/FR_EN/, 4: SRPRS/DE_EN/")
parser.add_argument("--mode", type=str, default="hybrid", help="\"hybrid\": Semantic + Lexical or \"sem\": Semantic")
parser.add_argument("--iters", type=int, default=1, help="Number of iteration for fusing similarity scores of entities and relations")
parser.add_argument("--tau", type=float, default=0.2)
parser.add_argument("--lam", type=float, default=0.5, help="Lambda for fusing")

args = parser.parse_args()
print(args)
mode = args.mode.lower()
data_path = "datas/" + ["DBP15k/zh_en/", "DBP15k/ja_en/", "DBP15k/fr_en/", "SRPRS/FR_EN/", "SRPRS/DE_EN/"][args.datapath]

refiner = Refiner(Data(data_path, mode), args.iters, args.tau, args.lam)

print(f"Refinement... iters: {args.iters}")
for i in range(args.iters+1):
    refiner.update_ent_sims(iters=i)
    refiner.update_rel_sims(iters=i)
    print(f"[iter {i}] ",end="")
    if i == 0 : 
        eval_sims(refiner.ent_sims, refiner.data)
        print()
        continue
    eval_sims(refiner.sinkhorn(refiner.ent_sims), refiner.data)
    print()

verifier = Verifier(refiner)
verify_ranks, verified_idx = verifier.verify()
print(f"[Verify] ",end="")
new_ranks, new_results = verifier.new_rank_results(verify_ranks) 
verified_case = verifier.verified_dist(new_results, verified_idx)