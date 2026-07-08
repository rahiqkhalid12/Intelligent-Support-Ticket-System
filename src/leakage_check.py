
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

TRAIN_PATH = "data/processed/train.csv"   # adjust path if needed
TEST_PATH  = "data/processed/test.csv"    # adjust path if needed

SIMILARITY_THRESHOLD = 0.97  # near-duplicate cutoff (1.0 = identical)
TOP_N_TO_SHOW = 15           # how many examples to print


def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df["text"] = train_df["text"].fillna("")
    test_df["text"] = test_df["text"].fillna("")

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    print("\nLoading SBERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding train texts...")
    train_emb = model.encode(
        train_df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    print("Encoding test texts...")
    test_emb = model.encode(
        test_df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    print("\nComputing test->train similarity (this may take a minute)...")
    # For each test ticket, find its single most similar train ticket
    max_sims = []
    best_train_idx = []

    chunk_size = 256
    for start in range(0, len(test_emb), chunk_size):
        end = min(start + chunk_size, len(test_emb))
        sims = util.cos_sim(test_emb[start:end], train_emb)  # [chunk, n_train]
        chunk_max, chunk_idx = sims.max(dim=1)
        max_sims.extend(chunk_max.cpu().numpy().tolist())
        best_train_idx.extend(chunk_idx.cpu().numpy().tolist())

    max_sims = np.array(max_sims)
    best_train_idx = np.array(best_train_idx)

    n_leaked = (max_sims >= SIMILARITY_THRESHOLD).sum()
    pct_leaked = n_leaked / len(test_df) * 100

    print("\n" + "=" * 60)
    print("LEAKAGE CHECK RESULTS")
    print("=" * 60)
    print(f"Threshold           : similarity >= {SIMILARITY_THRESHOLD}")
    print(f"Test tickets total  : {len(test_df)}")
    print(f"Near-duplicates     : {n_leaked} ({pct_leaked:.2f}%)")
    print(f"Max similarity      : {max_sims.max():.4f}")
    print(f"Mean similarity     : {max_sims.mean():.4f}")
    print(f"Median similarity   : {np.median(max_sims):.4f}")

    if n_leaked > 0:
        print(f"\nTop {min(TOP_N_TO_SHOW, n_leaked)} near-duplicate pairs:")
        leaked_idx = np.argsort(-max_sims)[:TOP_N_TO_SHOW]
        for i in leaked_idx:
            if max_sims[i] < SIMILARITY_THRESHOLD:
                break
            t_idx = best_train_idx[i]
            print(f"\n--- similarity {max_sims[i]:.4f} ---")
            print(f"TEST  [{i}]: {test_df.iloc[i]['text'][:150]}")
            print(f"TRAIN [{t_idx}]: {train_df.iloc[t_idx]['text'][:150]}")

    print("\n" + "=" * 60)
    if pct_leaked < 1.0:
        print("Result: Negligible leakage. RAG's strong scores are likely genuine.")
    elif pct_leaked < 5.0:
        print("Result: Some leakage present. Worth noting as a limitation in your report.")
    else:
        print("Result: Significant leakage. This likely explains RAG's high scores -")
        print("        consider re-splitting with near-duplicate-aware deduplication.")
    print("=" * 60)


if __name__ == "__main__":
    main()