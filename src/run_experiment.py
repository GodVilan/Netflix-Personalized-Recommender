"""
run_experiment.py
End-to-end experiment: trains ALS baseline + NCF + Two-Tower,
evaluates all three on MovieLens-1M, and runs A/B simulation.
Usage:
    python src/run_experiment.py --data_dir ml-1m/ --epochs 10
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

# Make sure src/ imports work when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from ab_testing import RecommendationExperiment
from data_processing import load_movielens, encode_ids, build_interaction_matrix, split_data, get_genre_features
from models import ALSModel, NCFModel, TwoTowerModel
from metrics import ndcg_at_k, recall_at_k, hit_rate_at_k, mrr_at_k
from trainer import train_neural_model, InteractionDataset, get_device


# ── A/B Simulation ───────────────────────────────────────────────────────────
def simulate_ab_test(n_rounds: int = 20000) -> dict:
    experiment = RecommendationExperiment(
        name="als_vs_two_tower_summer_2026",
        arms=["als_baseline", "two_tower"],
        mode="thompson",
        significance_level=0.05,
        min_samples_per_arm=1000,
    )
    rng = np.random.default_rng(42)
    true_ctrs = {"als_baseline": 0.12, "two_tower": 0.14}
    true_engagement = {"als_baseline": (420, 180), "two_tower": (470, 175)}
    for _ in range(n_rounds):
        arm = experiment.route()
        clicked = rng.random() < true_ctrs[arm]
        engagement = max(0, rng.normal(*true_engagement[arm])) if clicked else 0.0
        experiment.record(arm, clicked, engagement)
    return experiment.summary()


# ── Evaluation helper ──────────────────────────────────────────────────────────
def evaluate_model(model, val_df, n_users, n_items, k=10, device=None, model_type="neural"):
    """Evaluates a model on the validation set. Returns dict of metrics."""
    ndcgs, recalls, hits, mrrs = [], [], [], []

    # Sample up to 500 users for fast evaluation
    sample_users = val_df["user_idx"].unique()
    if len(sample_users) > 500:
        sample_users = np.random.choice(sample_users, 500, replace=False)

    for user in sample_users:
        user_val = val_df[val_df["user_idx"] == user]
        if len(user_val) == 0:
            continue
        true_items = user_val["item_idx"].values.tolist()

        if model_type == "als":
            scores = model.recommend(user, n_items)
        else:
            # Neural models: score all items
            model.eval()
            with torch.no_grad():
                user_tensor = torch.tensor([user] * n_items, dtype=torch.long)
                item_tensor = torch.tensor(list(range(n_items)), dtype=torch.long)
                if device:
                    user_tensor = user_tensor.to(device)
                    item_tensor = item_tensor.to(device)
                scores = model.score(user_tensor, item_tensor).cpu().numpy()

        top_k = np.argsort(scores)[::-1][:k].tolist()
        ndcgs.append(ndcg_at_k(true_items, top_k, k))
        recalls.append(recall_at_k(true_items, top_k, k))
        hits.append(hit_rate_at_k(true_items, top_k, k))
        mrrs.append(mrr_at_k(true_items, top_k, k))

    return {
        f"NDCG@{k}": round(float(np.mean(ndcgs)), 4),
        f"Recall@{k}": round(float(np.mean(recalls)), 4),
        f"HitRate@{k}": round(float(np.mean(hits)), 4),
        f"MRR@{k}": round(float(np.mean(mrrs)), 4),
    }


# ── Main Pipeline ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="ml-1m/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_factors", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ab_rounds", type=int, default=20000)
    parser.add_argument("--skip_ab", action="store_true", help="Skip A/B simulation")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Netflix Recommendation System — Experiment Pipeline")
    print("="*60)

    # ── Device ──
    device = get_device()

    # ── Step 1: A/B Simulation ──
    if not args.skip_ab:
        print("\n[1/4] Running A/B test simulation (Thompson Sampling)...")
        ab_results = simulate_ab_test(n_rounds=args.ab_rounds)
        print(json.dumps(ab_results, indent=2))
        ctr = ab_results["significance_tests"]["ctr"]
        eng = ab_results["significance_tests"]["engagement"]
        print(f"\n✔ CTR lift: {ctr['relative_lift_pct']:.2f}%  (p={ctr['p_value']}, significant={ctr['significant']})")
        print(f"✔ Engagement lift: {eng['arm_b_mean_engagement'] - eng['arm_a_mean_engagement']:.1f}s  (Cohen\'s d={eng['cohens_d']:.3f})")

    # ── Step 2: Load Data ──
    print("\n[2/4] Loading MovieLens-1M data...")
    ratings_path = os.path.join(args.data_dir, "ratings.dat")
    movies_path  = os.path.join(args.data_dir, "movies.dat")

    if not os.path.exists(ratings_path):
        print(f"  ❌ ERROR: ratings.dat not found at {ratings_path}")
        print(f"  Make sure ml-1m/ folder is in your project root.")
        sys.exit(1)

    ratings, movies = load_movielens(ratings_path, movies_path)
    ratings, user_enc, item_enc = encode_ids(ratings)
    n_users = ratings["user_idx"].nunique()
    n_items = ratings["item_idx"].nunique()
    print(f"  Users: {n_users:,} | Items: {n_items:,} | Ratings: {len(ratings):,}")

    train_df, val_df, test_df = split_data(ratings)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    genre_features, genre_names = get_genre_features(movies, item_enc)
    print(f"  Genres: {len(genre_names)} → {genre_names}")

    interaction_matrix = build_interaction_matrix(ratings, n_users, n_items)

    # Seen items per user (for negative sampling)
    seen_items = ratings.groupby("user_idx")["item_idx"].apply(set).to_dict()

    # ── Step 3: Train Models ──
    print("\n[3/4] Training models...")
    results = {}

    # ── 3a: ALS ──
    print("\n--- ALS Baseline ---")
    als = ALSModel(n_factors=args.n_factors, n_iterations=15, regularization=0.01)
    als.fit(interaction_matrix)
    results["ALS"] = evaluate_model(als, val_df, n_users, n_items, model_type="als")
    print(f"  ALS → {results['ALS']}")

    # Dataset for neural models
    train_dataset = InteractionDataset(
        user_ids=train_df["user_idx"].values,
        item_ids=train_df["item_idx"].values,
        n_items=n_items,
        n_neg=4,
        seen_items=seen_items,
        genre_features=genre_features,
    )

    # ── 3b: NCF ──
    print("\n--- NCF ---")
    ncf = NCFModel(n_users=n_users, n_items=n_items, n_factors=args.n_factors)
    val_fn_ncf = lambda m: evaluate_model(m, val_df, n_users, n_items, device=device)["NDCG@10"]
    ncf = train_neural_model(
        model=ncf, train_dataset=train_dataset, val_fn=val_fn_ncf,
        model_name="NCF", epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, use_mlflow=False,
    )
    results["NCF"] = evaluate_model(ncf, val_df, n_users, n_items, device=device)
    print(f"  NCF → {results['NCF']}")

    # ── 3c: Two-Tower ──
    print("\n--- Two-Tower ---")
    two_tower = TwoTowerModel(
        n_users=n_users, n_items=n_items,
        n_factors=args.n_factors, n_genres=len(genre_names)
    )
    val_fn_tt = lambda m: evaluate_model(m, val_df, n_users, n_items, device=device)["NDCG@10"]
    two_tower = train_neural_model(
        model=two_tower, train_dataset=train_dataset, val_fn=val_fn_tt,
        model_name="TwoTower", epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, use_mlflow=False,
    )
    results["TwoTower"] = evaluate_model(two_tower, val_df, n_users, n_items, device=device)
    print(f"  TwoTower → {results['TwoTower']}")

    # ── Step 4: Results Table ──
    print("\n[4/4] Final Results")
    print("\n" + "="*60)
    header = f"{'Model':<12} {'NDCG@10':>10} {'Recall@10':>10} {'HitRate@10':>12} {'MRR@10':>8}"
    print(header)
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<12} {metrics['NDCG@10']:>10} {metrics['Recall@10']:>10} {metrics['HitRate@10']:>12} {metrics['MRR@10']:>8}")
    print("="*60)

    # Save results
    output = {"models": results, "ab_test": ab_results if not args.skip_ab else {}}
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n✔ Results saved to results.json")

    # Save model checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(ncf.state_dict(), "checkpoints/ncf.pt")
    torch.save(two_tower.state_dict(), "checkpoints/two_tower.pt")
    print("✔ Model checkpoints saved to checkpoints/")
    print("\n✔ API server: uvicorn api.main:app --reload --port 8000")
    print("\nDone.")


if __name__ == "__main__":
    main()
