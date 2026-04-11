"""
run_experiment.py  —  DARE-Rec full experiment (2026-04-10, v4.1)

Pipeline:
  1. Load MovieLens-1M, 80/20 holdout split
  2. Fit TemporalEASE (closed-form, ~30s)
  3. Fit ImplicitALS  (~3 min)
  4. Train LightGCN   (~15-20 min on A100)
  5. DAREnsemble: grid search α/β/γ on val NDCG@10
  6. MMRReranker: genre-diverse re-ranking on ensemble scores
  7. Full evaluation: NDCG@10, Recall@10, HitRate@10, ILD@10, Coverage, Novelty
  8. Save checkpoints + score matrices + results.json + item_metadata.json

Usage:
  python src/run_experiment.py --data_dir ml-1m/
  python src/run_experiment.py --data_dir ml-1m/ --skip_ab

New in v4.1 (2026-04-10):
  Fixed os.makedirs("checkpoints") placement — must run BEFORE np.save.
  Previously caused FileNotFoundError on first run when checkpoints/ absent.

New in v4 (2026-04-10):
  After training LightGCN, materialize and save the PROPAGATED embeddings:
    checkpoints/lgcn_item_emb.npy  — (n_items, embed_dim) L2-normalized
    checkpoints/lgcn_user_emb.npy  — (n_users, embed_dim) L2-normalized

  These are E_final = mean(E0..EL) from lgcn.propagate(), which requires
  norm_adj to be live (it is, immediately after training). The API loads
  these files directly instead of re-running propagate() without a graph.

New in v3 (2026-04-10):
  Saves scores_ease.npy, scores_ials.npy, scores_lgcn.npy, scores_dare.npy.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from ab_testing      import RecommendationExperiment
from data_processing import (
    load_movielens, encode_ids,
    build_interaction_matrix, build_temporal_interaction_matrix,
    split_data_holdout, get_genre_features, get_item_metadata,
)
from models import (
    TemporalEASE, ImplicitALS, LightGCN, DAREnsemble, MMRReranker,
)
from metrics import (
    ndcg_at_k, recall_at_k, hit_rate_at_k, mrr,
    intra_list_diversity, evaluate_recommendations,
)
from trainer import train_lightgcn, BPRDataset, get_device


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return (mat / norms).astype(np.float32)


def scores_to_recs(score_matrix: np.ndarray, k: int = 10) -> dict:
    recs = {}
    for uid in range(len(score_matrix)):
        top_k = np.argsort(score_matrix[uid])[::-1][:k].tolist()
        recs[uid] = top_k
    return recs


def evaluate_score_matrix(
    score_matrix, ground_truth, genre_matrix,
    item_popularity, n_items, k=10, label="",
) -> dict:
    recs = scores_to_recs(score_matrix, k=k)
    metrics = evaluate_recommendations(
        recommendations=recs,
        ground_truth=ground_truth,
        k_values=[10],
        item_popularity=item_popularity,
        n_items=n_items,
        genre_matrix=genre_matrix,
    )
    if label:
        print(f"  {label:18s} | NDCG@10={metrics.get('NDCG@10',0):.4f}  "
              f"Recall@10={metrics.get('Recall@10',0):.4f}  "
              f"ILD@10={metrics.get('ILD@10',0):.4f}  "
              f"Coverage={metrics.get('Coverage',0):.4f}")
    return metrics


def simulate_ab_test(n_rounds: int = 20000) -> dict:
    experiment = RecommendationExperiment(
        name="ease_vs_dare_rec_2026",
        arms=["ease_baseline", "dare_rec"],
        mode="thompson",
        significance_level=0.05,
        min_samples_per_arm=1000,
    )
    rng = np.random.default_rng(42)
    true_ctrs       = {"ease_baseline": 0.12, "dare_rec": 0.155}
    true_engagement = {"ease_baseline": (420, 180), "dare_rec": (490, 175)}
    for _ in range(n_rounds):
        arm     = experiment.route()
        clicked = rng.random() < true_ctrs[arm]
        eng     = max(0, rng.normal(*true_engagement[arm])) if clicked else 0.0
        experiment.record(arm, clicked, eng)
    return experiment.summary()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="ml-1m/")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--lgcn_lr",     type=float, default=1e-3)
    parser.add_argument("--lgcn_dim",    type=int,   default=64)
    parser.add_argument("--lgcn_layers", type=int,   default=3)
    parser.add_argument("--mmr_lambda",  type=float, default=0.7)
    parser.add_argument("--skip_ab",     action="store_true")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("DARE-Rec — Diversity-Aware Re-ranking Ensemble Recommender")
    print("MovieLens-1M  |  80/20 Holdout  |  Full Catalog Eval")
    print("="*60)

    device = get_device()

    ab_results = {}
    if not args.skip_ab:
        print("\n[1/7] A/B test simulation (Thompson Sampling)...")
        ab_results = simulate_ab_test()
        ctr = ab_results["significance_tests"]["ctr"]
        print(f"  CTR lift: {ctr['relative_lift_pct']:.2f}%  (p={ctr['p_value']}, significant={ctr['significant']})")

    print("\n[2/7] Loading MovieLens-1M...")
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

    train_df, val_df, test_df = split_data_holdout(ratings, test_ratio=0.2, seed=42)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"  Split: 80/20 holdout (matches Anelli et al. 2022 protocol)")

    genre_matrix, genre_names = get_genre_features(movies, item_enc)
    print(f"  Genres: {len(genre_names)}")

    item_metadata = get_item_metadata(movies, item_enc)

    test_gt         = test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    val_gt          = val_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    item_popularity = train_df["item_idx"].value_counts().to_dict()
    seen_items      = train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    train_binary   = build_interaction_matrix(train_df, n_users, n_items)
    train_temporal = build_temporal_interaction_matrix(train_df, n_users, n_items, decay=0.001)

    print("\n[3/7] TemporalEASE...")
    ease   = TemporalEASE(l2_lambda=400.0)
    ease.fit(train_temporal)
    S_ease = ease.score_all_users(exclude_train=True)
    print(f"  Score matrix: {S_ease.shape}")

    print("\n[4/7] ImplicitALS...")
    ials = ImplicitALS(n_factors=256, regularization=0.01, iterations=50, alpha=1.0)
    ials.fit(train_binary)
    S_ials = ials.score_all_users()
    print(f"  Score matrix: {S_ials.shape}")

    print("\n[5/7] LightGCN...")
    lgcn = LightGCN(
        n_users=n_users, n_items=n_items,
        embed_dim=args.lgcn_dim, n_layers=args.lgcn_layers,
    )
    lgcn.build_graph(train_binary, device)

    bpr_dataset = BPRDataset(
        user_ids=train_df["user_idx"].values,
        item_ids=train_df["item_idx"].values,
        n_items=n_items,
        seen_items=seen_items,
        n_neg=8,
    )

    def val_fn_lgcn(m):
        S = m.score_all_users(train_binary, batch_size=512)
        val_users = list(val_gt.keys())[:1000]
        ndcgs = [
            ndcg_at_k(np.argsort(S[u])[::-1][:10].tolist(), val_gt[u], 10)
            for u in val_users if u in val_gt
        ]
        return float(np.mean(ndcgs))

    lgcn = train_lightgcn(
        model=lgcn, train_dataset=bpr_dataset, val_fn=val_fn_lgcn,
        epochs=args.epochs, batch_size=4096, lr=args.lgcn_lr,
        n_neg=8, device=device,
    )
    S_lgcn = lgcn.score_all_users(train_binary, batch_size=512)
    print(f"  Score matrix: {S_lgcn.shape}")

    # Materialize propagated embeddings while norm_adj is still live
    print("  [LightGCN] materializing propagated embeddings...")
    lgcn.eval()
    with torch.no_grad():
        u_emb_prop, i_emb_prop = lgcn.propagate()
        u_emb_np = u_emb_prop.cpu().numpy().astype(np.float32)
        i_emb_np = i_emb_prop.cpu().numpy().astype(np.float32)

    # Sanity check
    unit_i = l2_normalize(i_emb_np)
    sample_idx = np.random.default_rng(42).choice(len(unit_i), size=200, replace=False)
    sub  = unit_i[sample_idx]
    gram = sub @ sub.T
    np.fill_diagonal(gram, 0)
    med_cos = float(np.median(gram[gram != 0]))
    print(f"  [LightGCN] propagated embeddings median pairwise cosine: {med_cos:.4f}")
    if med_cos > 0.5:
        print(f"  WARNING: median cosine={med_cos:.4f} still high. Consider more epochs or higher n_neg.")
    else:
        print(f"  Embedding space looks healthy.")

    # makedirs BEFORE save (v4.1 fix)
    os.makedirs("checkpoints", exist_ok=True)

    lgcn_item_emb = l2_normalize(i_emb_np)
    lgcn_user_emb = l2_normalize(u_emb_np)
    np.save("checkpoints/lgcn_item_emb.npy", lgcn_item_emb)
    np.save("checkpoints/lgcn_user_emb.npy", lgcn_user_emb)
    print(f"  Saved lgcn_item_emb.npy {lgcn_item_emb.shape}  lgcn_user_emb.npy {lgcn_user_emb.shape}")

    print("\n[6/7] DAREnsemble (learned score fusion)...")
    ensemble   = DAREnsemble()
    ensemble.fit(S_ease=S_ease, S_lgcn=S_lgcn, S_ials=S_ials, val_ground_truth=val_gt)
    S_ensemble = ensemble.predict(S_ease, S_lgcn, S_ials)

    print("\n[7/7] MMR Re-ranking (diversity-aware)...")
    mmr_reranker = MMRReranker(lambda_=args.mmr_lambda, k=10)
    mmr_reranker.build_genre_sim(genre_matrix)
    mmr_recs = mmr_reranker.rerank_all(S_ensemble, n_candidates=50)

    print("\n" + "="*60)
    print("Test Results  (80/20 holdout, all 6040 users, full catalog)")
    print("="*60)

    results = {}
    for label, S in [
        ("TemporalEASE", S_ease),
        ("ImplicitALS",  S_ials),
        ("LightGCN",     S_lgcn),
        ("DAREnsemble",  S_ensemble),
    ]:
        results[label] = evaluate_score_matrix(
            S, test_gt, genre_matrix, item_popularity, n_items, label=label
        )

    mmr_metrics = evaluate_recommendations(
        recommendations=mmr_recs, ground_truth=test_gt, k_values=[10],
        item_popularity=item_popularity, n_items=n_items, genre_matrix=genre_matrix,
    )
    results["DARE+MMR"] = mmr_metrics
    m = mmr_metrics
    print(f"  {'DARE+MMR':18s} | NDCG@10={m.get('NDCG@10',0):.4f}  "
          f"Recall@10={m.get('Recall@10',0):.4f}  "
          f"ILD@10={m.get('ILD@10',0):.4f}  "
          f"Coverage={m.get('Coverage',0):.4f}")

    print("="*60)
    print("\nPublished baselines (Anelli et al. 2022, same protocol):")
    print("  EASE^R (standard)  NDCG@10=0.336")
    print("  iALS               NDCG@10=0.306")
    print("  NeuMF              NDCG@10=0.277")

    # Save all checkpoints
    torch.save(lgcn.state_dict(), "checkpoints/lightgcn.pt")
    np.savez("checkpoints/ease_B.npz", B=ease.B)
    np.savez(
        "checkpoints/ials_factors.npz",
        user_factors=np.asarray(ials.model.user_factors, dtype=np.float32),
        item_factors=np.asarray(ials.model.item_factors, dtype=np.float32),
    )
    np.save("checkpoints/genre_matrix.npy", genre_matrix)
    np.save("checkpoints/scores_ease.npy", S_ease.astype(np.float32))
    np.save("checkpoints/scores_ials.npy", S_ials.astype(np.float32))
    np.save("checkpoints/scores_lgcn.npy", S_lgcn.astype(np.float32))
    np.save("checkpoints/scores_dare.npy", S_ensemble.astype(np.float32))

    with open("checkpoints/ensemble_weights.json", "w") as f:
        json.dump({"alpha": ensemble.alpha, "beta": ensemble.beta, "gamma": ensemble.gamma}, f, indent=2)

    with open("checkpoints/item_metadata.json", "w") as f:
        json.dump(item_metadata, f)
    print(f"  Saved item_metadata.json ({len(item_metadata):,} titles)")

    output = {
        "models": results,
        "ensemble_weights": {
            "alpha_ease":    ensemble.alpha,
            "beta_lightgcn": ensemble.beta,
            "gamma_ials":    ensemble.gamma,
        },
        "ab_test": ab_results,
    }
    for path in ["results.json", "checkpoints/metrics.json"]:
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    print("\n✔ Saved checkpoints:")
    print("    lightgcn.pt, ease_B.npz, ials_factors.npz, genre_matrix.npy")
    print("    lgcn_item_emb.npy, lgcn_user_emb.npy")
    print("    scores_ease.npy, scores_ials.npy, scores_lgcn.npy, scores_dare.npy")
    print("    ensemble_weights.json, item_metadata.json, metrics.json")
    print("✔ Start API:  uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print("Done.")


if __name__ == "__main__":
    main()
