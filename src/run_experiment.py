"""
run_experiment.py
End-to-end experiment: trains ALS + NCF + Two-Tower on MovieLens-1M,
evaluates all three, and optionally runs an A/B simulation.

Usage:
    python src/run_experiment.py --data_dir ml-1m/ --epochs 30
    python src/run_experiment.py --data_dir ml-1m/ --epochs 30 --skip_ab
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


# ── A/B Simulation ────────────────────────────────────────────────────────────
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


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(
    model,
    test_df,
    seen_items: dict,           # {user_idx: set of train item indices}
    genre_matrix: np.ndarray,
    n_items: int,
    k: int = 10,
    device=None,
    model_type: str = "neural",
    n_eval_users: int = 500,
) -> dict:
    """
    Consistent evaluation for all model types:
      1. Uses Leave-One-Out test set (one ground-truth item per user)
      2. Excludes seen (training) items from ranking for all models
      3. Neural models score ALL items then exclude seen — same as ALS
    """
    ndcgs, recalls, hits, mrrs = [], [], [], []

    all_users = test_df["user_idx"].unique()
    if len(all_users) > n_eval_users:
        rng = np.random.default_rng(0)
        all_users = rng.choice(all_users, n_eval_users, replace=False)

    all_items_tensor = None
    if model_type == "neural" and device is not None:
        all_items_tensor = torch.arange(n_items, dtype=torch.long, device=device)

    for user in all_users:
        user_test = test_df[test_df["user_idx"] == user]
        if len(user_test) == 0:
            continue
        true_items  = set(user_test["item_idx"].values.tolist())
        seen        = seen_items.get(int(user), set())
        unseen_mask = np.ones(n_items, dtype=bool)
        unseen_mask[list(seen)] = False

        if model_type == "als":
            # ALS: pass seen items to exclude directly
            seen_arr = np.array(list(seen), dtype=np.int64) if seen else None
            top_k = model.recommend(int(user), n=k, exclude_seen=seen_arr).tolist()

        else:
            # Neural: score all items, zero out seen, take top-k
            model.eval()
            with torch.no_grad():
                user_t = torch.full((n_items,), int(user), dtype=torch.long, device=device)
                scores = model.score(user_t, all_items_tensor).cpu().numpy()
            scores[list(seen)] = -np.inf       # exclude seen items
            top_k = np.argsort(scores)[::-1][:k].tolist()

        ndcgs.append(ndcg_at_k(top_k, true_items, k))
        recalls.append(recall_at_k(top_k, true_items, k))
        hits.append(hit_rate_at_k(top_k, true_items, k))
        mrrs.append(mrr(top_k, true_items))

    return {
        f"NDCG@{k}":    round(float(np.mean(ndcgs)),   4),
        f"Recall@{k}": round(float(np.mean(recalls)), 4),
        f"HitRate@{k}": round(float(np.mean(hits)),    4),
        "MRR":          round(float(np.mean(mrrs)),    4),
    }


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="ml-1m/")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--n_factors",  type=int,   default=128)
    parser.add_argument("--batch_size", type=int,   default=2048)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--n_neg",      type=int,   default=64,
                        help="Negatives per positive (64+ recommended for InfoNCE)")
    parser.add_argument("--n_eval_users", type=int, default=500)
    parser.add_argument("--ab_rounds",  type=int,   default=20000)
    parser.add_argument("--skip_ab",    action="store_true")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Netflix Recommendation System — Experiment Pipeline")
    print("="*60)

    device = get_device()

    # ── A/B Simulation ──
    ab_results = {}
    if not args.skip_ab:
        print("\n[1/4] Running A/B test simulation (Thompson Sampling)...")
        ab_results = simulate_ab_test(n_rounds=args.ab_rounds)
        ctr = ab_results["significance_tests"]["ctr"]
        eng = ab_results["significance_tests"]["engagement"]
        print(f"  CTR lift: {ctr['relative_lift_pct']:.2f}%  (p={ctr['p_value']}, significant={ctr['significant']})")
        print(f"  Engagement lift: {eng['arm_b_mean_engagement'] - eng['arm_a_mean_engagement']:.1f}s  (Cohen's d={eng['cohens_d']:.3f})")

    # ── Load Data ──
    print("\n[2/4] Loading MovieLens-1M data...")
    ratings_path = os.path.join(args.data_dir, "ratings.dat")
    movies_path  = os.path.join(args.data_dir, "movies.dat")

    if not os.path.exists(ratings_path):
        print(f"  ERROR: ratings.dat not found at {ratings_path}")
        sys.exit(1)

    ratings, movies      = load_movielens(ratings_path, movies_path)
    ratings, user_enc, item_enc = encode_ids(ratings)
    n_users = ratings["user_idx"].nunique()
    n_items = ratings["item_idx"].nunique()
    print(f"  Users: {n_users:,} | Items: {n_items:,} | Ratings: {len(ratings):,}")

    # Leave-one-out split
    train_df, val_df, test_df = split_data(ratings)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"  Split: Leave-One-Out (timestamp-based — no temporal leakage)")

    genre_features, genre_names = get_genre_features(movies, item_enc)
    n_genres = len(genre_names)
    print(f"  Genres: {n_genres} → {genre_names}")

    # Build interaction matrix from TRAIN only (not val/test)
    interaction_matrix = build_interaction_matrix(train_df, n_users, n_items)

    # Seen items per user (train only) — used to exclude at eval time
    seen_items = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    # ── Train Models ──
    print("\n[3/4] Training models...")
    results = {}

    # ── ALS ──
    print("\n--- ALS Baseline (CollaborativeFilteringALS) ---")
    print(f"  n_factors={args.n_factors}, iterations=30, alpha=1.0, regularization=0.05")
    als = CollaborativeFilteringALS(
        n_factors=args.n_factors,
        iterations=30,
        regularization=0.05,
        alpha=1.0,            # ML-1M has explicit ratings — low alpha avoids over-confidence
    )
    als.fit(interaction_matrix)
    results["ALS"] = evaluate_model(
        als, test_df, seen_items, genre_features,
        n_items, model_type="als", n_eval_users=args.n_eval_users
    )
    print(f"  ALS → {results['ALS']}")

    # Shared dataset for neural models
    train_dataset = InteractionDataset(
        user_ids=train_df["user_idx"].values,
        item_ids=train_df["item_idx"].values,
        n_items=n_items,
        n_neg=args.n_neg,            # 64 (was 4 — 16x more negatives)
        seen_items=seen_items,
        genre_features=genre_features,
    )

    # ── NCF ──
    print("\n--- NCF (NeuralMatrixFactorization) ---")
    ncf = NeuralMatrixFactorization(
        n_users=n_users, n_items=n_items, mf_dim=args.n_factors
    )
    val_fn_ncf = lambda m: evaluate_model(
        m, val_df, seen_items, genre_features,
        n_items, device=device, model_type="neural", n_eval_users=200
    )[f"NDCG@10"]
    ncf = train_neural_model(
        model=ncf, train_dataset=train_dataset, val_fn=val_fn_ncf,
        model_name="NCF", epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, use_mlflow=False,
    )
    results["NCF"] = evaluate_model(
        ncf, test_df, seen_items, genre_features,
        n_items, device=device, model_type="neural", n_eval_users=args.n_eval_users
    )
    print(f"  NCF → {results['NCF']}")

    # ── Two-Tower ──
    print("\n--- Two-Tower (TwoTowerRetrieval) ---")
    two_tower = TwoTowerRetrieval(
        n_users=n_users, n_items=n_items,
        n_genres=n_genres, embed_dim=args.n_factors,
        genre_features=genre_features,  # stored as buffer for inference
    )
    val_fn_tt = lambda m: evaluate_model(
        m, val_df, seen_items, genre_features,
        n_items, device=device, model_type="neural", n_eval_users=200
    )[f"NDCG@10"]
    two_tower = train_neural_model(
        model=two_tower, train_dataset=train_dataset, val_fn=val_fn_tt,
        model_name="TwoTower", epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, use_mlflow=False,
    )
    results["TwoTower"] = evaluate_model(
        two_tower, test_df, seen_items, genre_features,
        n_items, device=device, model_type="neural", n_eval_users=args.n_eval_users
    )
    print(f"  TwoTower → {results['TwoTower']}")

    # ── Results Table ──
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
    print("✔ Checkpoints saved to checkpoints/")
    print("\n✔ API server: uvicorn api.main:app --reload --port 8000")
    print("\nDone.")


if __name__ == "__main__":
    main()
