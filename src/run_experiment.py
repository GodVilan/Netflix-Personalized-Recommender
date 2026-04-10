"""
run_experiment.py
End-to-end experiment: trains ALS + NCF + Two-Tower on MovieLens-1M.

Usage:
    python src/run_experiment.py --data_dir ml-1m/ --epochs 30
    python src/run_experiment.py --data_dir ml-1m/ --epochs 30 --skip_ab

Fix log:
  Fix-10 (2026-04-10):
    ALS:      alpha default 1.0 → 40. Propagated to print statement.
    NCF:      lr 5e-4 → 1e-3. n_neg 64 → 200. accum_steps 1 → 2.
              mf_dim passed as 128 (was implicitly n_factors=256 before).
              mlp_dims passed as [512,256,128].
    TwoTower: lr 1e-3 → 3e-4. accum_steps 2 → 1.
              embed_dim and tower_dims come from model defaults (128, [256]).

  Fix-9 (2026-04-07): Full evaluation (all 6040 users) on test set.
  Fix-8 (2026-04-07): TwoTower batch 4096, lr 1e-3, accum 2.
  Fix-7 (2026-04-07): NCF heavy embedding L2.
  Fix-6 (2026-04-07): ALS → implicit library.
  Fix-1–5 (2026-04-06): NCF arch fixes, ALS hyperparams.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from ab_testing       import RecommendationExperiment
from data_processing  import load_movielens, encode_ids, build_interaction_matrix, split_data, get_genre_features
from models           import CollaborativeFilteringALS, NeuralMatrixFactorization, TwoTowerRetrieval
from metrics          import ndcg_at_k, recall_at_k, hit_rate_at_k, mrr
from trainer          import train_neural_model, InteractionDataset, get_device


def simulate_ab_test(n_rounds: int = 20000) -> dict:
    experiment = RecommendationExperiment(
        name="als_vs_two_tower_2026",
        arms=["als_baseline", "two_tower"],
        mode="thompson",
        significance_level=0.05,
        min_samples_per_arm=1000,
    )
    rng = np.random.default_rng(42)
    true_ctrs       = {"als_baseline": 0.12, "two_tower": 0.14}
    true_engagement = {"als_baseline": (420, 180), "two_tower": (470, 175)}
    for _ in range(n_rounds):
        arm        = experiment.route()
        clicked    = rng.random() < true_ctrs[arm]
        engagement = max(0, rng.normal(*true_engagement[arm])) if clicked else 0.0
        experiment.record(arm, clicked, engagement)
    return experiment.summary()


def evaluate_model(
    model,
    test_df,
    seen_items: dict,
    genre_matrix: np.ndarray,
    n_items: int,
    k: int = 10,
    device=None,
    model_type: str = "neural",
    n_eval_users: int = None,
) -> dict:
    ndcgs, recalls, hits, mrrs = [], [], [], []
    all_users = test_df["user_idx"].unique()
    if n_eval_users is not None and len(all_users) > n_eval_users:
        rng = np.random.default_rng(0)
        all_users = rng.choice(all_users, n_eval_users, replace=False)

    all_items_tensor = None
    if model_type == "neural" and device is not None:
        all_items_tensor = torch.arange(n_items, dtype=torch.long, device=device)

    for user in all_users:
        user_test = test_df[test_df["user_idx"] == user]
        if len(user_test) == 0:
            continue
        true_items = set(user_test["item_idx"].values.tolist())
        seen       = seen_items.get(int(user), set())

        if model_type == "als":
            seen_arr = np.array(list(seen), dtype=np.int64) if seen else None
            top_k = model.recommend(int(user), n=k, exclude_seen=seen_arr).tolist()
        else:
            model.eval()
            with torch.no_grad():
                user_t = torch.full((n_items,), int(user), dtype=torch.long, device=device)
                scores = model.score(user_t, all_items_tensor).cpu().numpy()
            scores[list(seen)] = -np.inf
            top_k = np.argsort(scores)[::-1][:k].tolist()

        ndcgs.append(ndcg_at_k(top_k, true_items, k))
        recalls.append(recall_at_k(top_k, true_items, k))
        hits.append(hit_rate_at_k(top_k, true_items, k))
        mrrs.append(mrr(top_k, true_items))

    return {
        f"NDCG@{k}":     round(float(np.mean(ndcgs)),   4),
        f"Recall@{k}":  round(float(np.mean(recalls)), 4),
        f"HitRate@{k}": round(float(np.mean(hits)),    4),
        "MRR":           round(float(np.mean(mrrs)),    4),
        "n_users_eval":  len(ndcgs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="ml-1m/")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--n_factors",   type=int,   default=256,
                        help="ALS n_factors and TwoTower embed_dim base")
    parser.add_argument("--ncf_mf_dim",  type=int,   default=128,
                        help="NCF GMF embedding dim (Fix-10: was n_factors=256)")
    parser.add_argument("--ncf_batch",   type=int,   default=2048)
    parser.add_argument("--tt_batch",    type=int,   default=4096)
    parser.add_argument("--ncf_lr",      type=float, default=1e-3,    # Fix-10: was 5e-4
                        help="NCF learning rate")
    parser.add_argument("--tt_lr",       type=float, default=3e-4,    # Fix-10: was 1e-3
                        help="TwoTower learning rate")
    parser.add_argument("--n_neg",       type=int,   default=200,     # Fix-10: was 64
                        help="Negatives per positive for NCF")
    parser.add_argument("--ncf_accum",   type=int,   default=2,       # Fix-10: was 1
                        help="Gradient accumulation for NCF")
    parser.add_argument("--tt_accum",    type=int,   default=1,       # Fix-10: was 2
                        help="Gradient accumulation for TwoTower (1=no accum; accum breaks in-batch neg semantics)")
    parser.add_argument("--als_alpha",   type=float, default=40.0,    # Fix-10: was 1.0
                        help="ALS confidence weight alpha (Hu et al. 2008)")
    parser.add_argument("--ab_rounds",   type=int,   default=20000)
    parser.add_argument("--skip_ab",     action="store_true")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Netflix Recommendation System — Experiment Pipeline")
    print("="*60)

    device = get_device()

    ab_results = {}
    if not args.skip_ab:
        print("\n[1/4] Running A/B test simulation (Thompson Sampling)...")
        ab_results = simulate_ab_test(n_rounds=args.ab_rounds)
        ctr = ab_results["significance_tests"]["ctr"]
        eng = ab_results["significance_tests"]["engagement"]
        print(f"  CTR lift: {ctr['relative_lift_pct']:.2f}%  (p={ctr['p_value']}, significant={ctr['significant']})")
        print(f"  Engagement lift: {eng['arm_b_mean_engagement'] - eng['arm_a_mean_engagement']:.1f}s  (Cohen's d={eng['cohens_d']:.3f})")

    print("\n[2/4] Loading MovieLens-1M data...")
    ratings_path = os.path.join(args.data_dir, "ratings.dat")
    movies_path  = os.path.join(args.data_dir, "movies.dat")
    if not os.path.exists(ratings_path):
        print(f"  ERROR: ratings.dat not found at {ratings_path}")
        sys.exit(1)

    ratings, movies             = load_movielens(ratings_path, movies_path)
    ratings, user_enc, item_enc = encode_ids(ratings)
    n_users = ratings["user_idx"].nunique()
    n_items = ratings["item_idx"].nunique()
    print(f"  Users: {n_users:,} | Items: {n_items:,} | Ratings: {len(ratings):,}")

    train_df, val_df, test_df = split_data(ratings)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"  Split: Leave-One-Out (timestamp-based — no temporal leakage)")

    genre_features, genre_names = get_genre_features(movies, item_enc)
    n_genres = len(genre_names)
    print(f"  Genres: {n_genres} → {genre_names}")

    interaction_matrix = build_interaction_matrix(train_df, n_users, n_items)
    seen_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    print("\n[3/4] Training models...")
    results = {}

    # ── ALS ──
    print("\n--- ALS Baseline (CollaborativeFilteringALS) ---")
    print(f"  n_factors={args.n_factors}, iterations=30, alpha={args.als_alpha}, regularization=0.01")
    print(f"  Backend: implicit.als.AlternatingLeastSquares (Conjugate Gradient)")
    als = CollaborativeFilteringALS(
        n_factors=args.n_factors,
        iterations=30,
        regularization=0.01,
        alpha=args.als_alpha,          # Fix-10: default now 40.0
    )
    als.fit(interaction_matrix)
    results["ALS"] = evaluate_model(
        als, test_df, seen_items, genre_features,
        n_items, model_type="als", n_eval_users=None
    )
    print(f"  ALS → {results['ALS']}")

    # Shared dataset
    train_dataset = InteractionDataset(
        user_ids=train_df["user_idx"].values,
        item_ids=train_df["item_idx"].values,
        n_items=n_items,
        n_neg=args.n_neg,              # Fix-10: default now 200
        seen_items=seen_items,
        genre_features=genre_features,
    )

    # ── NCF ──
    print("\n--- NCF (NeuralMatrixFactorization) ---")
    print(f"  mf_dim={args.ncf_mf_dim}, mlp_dims=[512,256,128], embed_wd=1e-2, embed_dropout=0.3")
    print(f"  lr={args.ncf_lr}, n_neg={args.n_neg}, accum={args.ncf_accum} → effective batch {args.ncf_batch * args.ncf_accum}")
    ncf = NeuralMatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        mf_dim=args.ncf_mf_dim,        # Fix-10: 128 (balanced GMF/MLP)
        mlp_dims=[512, 256, 128],       # Fix-10: wider MLP
    )
    val_fn_ncf = lambda m: evaluate_model(
        m, val_df, seen_items, genre_features,
        n_items, device=device, model_type="neural", n_eval_users=1000
    )["NDCG@10"]
    ncf = train_neural_model(
        model=ncf, train_dataset=train_dataset, val_fn=val_fn_ncf,
        model_name="NCF",
        epochs=args.epochs,
        batch_size=args.ncf_batch,
        lr=args.ncf_lr,                # Fix-10: 1e-3
        weight_decay=1e-5,
        accum_steps=args.ncf_accum,    # Fix-10: 2
        device=device,
        use_mlflow=False,
    )
    results["NCF"] = evaluate_model(
        ncf, test_df, seen_items, genre_features,
        n_items, device=device, model_type="neural", n_eval_users=None
    )
    print(f"  NCF → {results['NCF']}")

    # ── Two-Tower ──
    print("\n--- Two-Tower (TwoTowerRetrieval) ---")
    print(f"  IN-BATCH negatives | embed_dim=128 | tower_dims=[256]")
    print(f"  batch={args.tt_batch} | lr={args.tt_lr} | accum={args.tt_accum} | temp_init=0.07")
    two_tower = TwoTowerRetrieval(
        n_users=n_users,
        n_items=n_items,
        n_genres=n_genres,
        embed_dim=128,                 # Fix-10: was 256
        tower_dims=[256],              # Fix-10: was [512, 256]
        genre_features=genre_features,
        use_inbatch_negatives=True,
    )
    val_fn_tt = lambda m: evaluate_model(
        m, val_df, seen_items, genre_features,
        n_items, device=device, model_type="neural", n_eval_users=1000
    )["NDCG@10"]
    two_tower = train_neural_model(
        model=two_tower, train_dataset=train_dataset, val_fn=val_fn_tt,
        model_name="TwoTower",
        epochs=args.epochs,
        batch_size=args.tt_batch,
        lr=args.tt_lr,                 # Fix-10: 3e-4
        weight_decay=1e-5,
        accum_steps=args.tt_accum,     # Fix-10: 1
        device=device,
        use_mlflow=False,
    )
    results["TwoTower"] = evaluate_model(
        two_tower, test_df, seen_items, genre_features,
        n_items, device=device, model_type="neural", n_eval_users=None
    )
    print(f"  TwoTower → {results['TwoTower']}")

    print("\n[4/4] Final Results")
    print("\n" + "="*60)
    print(f"{'Model':<12} {'NDCG@10':>10} {'Recall@10':>10} {'HitRate@10':>12} {'MRR':>8}")
    print("-"*60)
    for name, m in results.items():
        print(f"{name:<12} {m['NDCG@10']:>10} {m['Recall@10']:>10} {m['HitRate@10']:>12} {m['MRR']:>8}")
    print("="*60)

    output = {"models": results, "ab_test": ab_results}
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n✔ Results saved to results.json")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(ncf.state_dict(),       "checkpoints/ncf.pt")
    torch.save(two_tower.state_dict(), "checkpoints/two_tower.pt")
    np.savez("checkpoints/als_factors.npz",
             user_factors=als.user_factors,
             item_factors=als.item_factors)
    np.save("checkpoints/genre_matrix.npy", genre_features)

    movie_map   = movies.set_index("movie_id")["title"].to_dict()
    encoded_ids = item_enc.classes_
    meta = {str(idx): movie_map.get(mid, f"Movie {mid}")
            for idx, mid in enumerate(encoded_ids)}
    with open("checkpoints/item_metadata.json", "w") as f:
        json.dump(meta, f)

    import datetime
    metrics_out = {
        "model": "two_tower_v3",
        "eval_date": datetime.date.today().isoformat(),
        "dataset": "MovieLens-1M",
        "metrics": results.get("TwoTower", {}),
        "models": results,
    }
    with open("checkpoints/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("✔ Checkpoints saved to checkpoints/")
    print("\n✔ API server: uvicorn api.main:app --reload --port 8000")
    print("\nDone.")


if __name__ == "__main__":
    main()
