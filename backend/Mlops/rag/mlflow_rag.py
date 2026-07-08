import json
import time
import warnings
import sys
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ==========================================================
# Paths — adjust if your folder structure differs
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models" / "rag"
MLFLOW_DB    = Path(__file__).resolve().parent / "mlflow.db"
REPORTS_DIR  = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# rag/ folder contains retrieve.py and generate.py
RAG_DIR = PROJECT_ROOT / "rag"
sys.path.insert(0, str(RAG_DIR))

# ==========================================================
# Configuration
# ==========================================================
TOP_K              = 5
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
RETRIEVER          = "FAISS"
VECTOR_DB          = "FAISS + Azure AI Search"

# Number of tickets to run FULL generation on (retrieval + Qwen response).
# Classification metrics are computed on the FULL test set using retrieval
# only (much faster). Generation is sampled to avoid HF rate limits.
EVAL_SAMPLE_SIZE   = 100   # set to None to evaluate all (very slow)

TARGETS = ["type", "queue", "priority"]

# ==========================================================
# MLflow setup
# ==========================================================
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("RAG_Models")

# ==========================================================
# Import existing RAG functions
# ==========================================================
def import_rag_modules():
    """Import retrieval only (no Qwen download)."""

    try:
        import retrieve as retrieve_mod

        retrieve_fn = retrieve_mod.retrieve

        print("Loaded retrieval from retrieve.py")

        return retrieve_fn

    except Exception as e:
        raise ImportError(
            f"Could not import retrieve.py\n{e}"
        )

# ==========================================================
# Weighted vote — same logic as evaluate.py
# ==========================================================
def weighted_vote(docs, label_key):
    votes = Counter()
    for doc in docs:
        votes[doc.get(label_key, "")] += doc.get("similarity_score", doc.get("score", 1.0))
    return votes.most_common(1)[0][0] if votes else "Unknown"

# ==========================================================
# Classification evaluation (full test set, retrieval only)
# ==========================================================
def evaluate_classification(test_df, retrieve_fn):
    print(f"\nClassification evaluation on {len(test_df)} test tickets...")

    true  = {t: [] for t in TARGETS}
    pred  = {t: [] for t in TARGETS}
    retrieval_times = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        if i % 100 == 0:
            print(f"  {i}/{len(test_df)} tickets processed...")

        query = str(row.get("text", ""))

        t0   = time.perf_counter()
        docs = retrieve_fn(query, top_k=TOP_K)
        retrieval_times.append(time.perf_counter() - t0)

        if not docs:
            continue

        for target in TARGETS:
            true[target].append(row[target])
            pred[target].append(weighted_vote(docs, target))

    metrics = {}
    reports = {}
    for target in TARGETS:
        if not true[target]:
            continue
        metrics[target] = {
            "accuracy":  round(float(accuracy_score(true[target], pred[target])), 4),
            "precision": round(float(precision_score(true[target], pred[target], average="weighted", zero_division=0)), 4),
            "recall":    round(float(recall_score(true[target], pred[target], average="weighted", zero_division=0)), 4),
            "f1":        round(float(f1_score(true[target], pred[target], average="weighted", zero_division=0)), 4),
        }
        reports[target] = classification_report(
            true[target], pred[target], zero_division=0
        )

    avg_retrieval_ms = round(sum(retrieval_times) / len(retrieval_times) * 1000, 2) if retrieval_times else 0.0
    print(f"  Average retrieval latency: {avg_retrieval_ms} ms")

    return metrics, reports, avg_retrieval_ms


# ==========================================================
# MLflow logging
# ==========================================================
def log_to_mlflow(
    test_df,
    classification_metrics,
    classification_reports,
    avg_retrieval_ms,
    avg_total_ms,
    sample_size,
):
    with mlflow.start_run(run_name="RAG_Qwen_SBERT"):

        # ── Parameters ────────────────────────────────────
        mlflow.log_param("embedding_model",   EMBEDDING_MODEL)
        mlflow.log_param("retriever",         RETRIEVER)
        mlflow.log_param("vector_db",         VECTOR_DB)
        mlflow.log_param("top_k",             TOP_K)
        mlflow.log_param("test_set_size",     len(test_df))
        mlflow.log_param("generation_sample", sample_size or len(test_df))
        mlflow.log_param("voting_method",     "similarity_weighted_majority_vote")

        # ── Classification metrics (per target) ───────────
        for target, metrics in classification_metrics.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{target}_{metric_name}", value)

        # ── Latency metrics ───────────────────────────────
        mlflow.log_metric("avg_retrieval_latency_ms",  avg_retrieval_ms)
        mlflow.log_metric("avg_total_latency_ms",      avg_total_ms)

        # ── Tags ──────────────────────────────────────────
        mlflow.set_tag("model_type",  "RAG")
        mlflow.set_tag("retriever",   RETRIEVER)
        mlflow.set_tag("embeddings",  EMBEDDING_MODEL)

        # ── Artifact 1: RAG config ─────────────────────────
        rag_config = {
            "embedding_model":   EMBEDDING_MODEL,
            "retriever":         RETRIEVER,
            "vector_db":         VECTOR_DB,
            "top_k":             TOP_K,
            "voting_method":     "similarity_weighted_majority_vote",
            "faiss_index_path":  str(MODELS_DIR / "faiss.index"),
            "documents_path":    str(MODELS_DIR / "documents.pkl"),
        }
        config_path = REPORTS_DIR / "rag_config.json"
        config_path.write_text(json.dumps(rag_config, indent=2))
        mlflow.log_artifact(str(config_path), artifact_path="config")

        # ── Artifact 2: Evaluation report ─────────────────
        eval_report = {
            "classification_metrics": classification_metrics,
            "latency_ms": {
                "avg_retrieval":  avg_retrieval_ms,
                "avg_total":      avg_total_ms,
            },
            "dataset": {
                "test_size":       len(test_df),
                "generation_sample": sample_size or len(test_df),
            },
        }
        report_path = REPORTS_DIR / "rag_evaluation_report.json"
        report_path.write_text(json.dumps(eval_report, indent=2))
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")

        # ── Artifact 3: Per-class classification reports ───
        for target, report_text in classification_reports.items():
            target_report_path = REPORTS_DIR / f"rag_{target}_classification_report.txt"
            target_report_path.write_text(report_text)
            mlflow.log_artifact(str(target_report_path), artifact_path="classification_reports")

        print("\nLogged to MLflow:")
        print(f"  Experiment : RAG_Models")
        print(f"  Run        : RAG_Qwen_SBERT")

# ==========================================================
# Main
# ==========================================================
def main():
    print("=" * 60)
    print("RAG MLflow Evaluation")
    print("=" * 60)

    # Load modules
    retrieve_fn = import_rag_modules()
    # Load test data
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_df["text"] = test_df["text"].fillna("")
    print(f"Test samples: {len(test_df)}")

    # Classification evaluation (full test set)
    classification_metrics, classification_reports, avg_retrieval_ms = \
        evaluate_classification(test_df, retrieve_fn)

    # Generation latency (sampled)
    avg_generation_ms = 0.0
    avg_total_ms = avg_retrieval_ms

    # Log to MLflow
    log_to_mlflow(
        test_df=test_df,
        classification_metrics=classification_metrics,
        classification_reports=classification_reports,
        avg_retrieval_ms=avg_retrieval_ms,
        avg_total_ms=avg_total_ms,
        sample_size=EVAL_SAMPLE_SIZE,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    for target, metrics in classification_metrics.items():
        print(f"\n{target.upper()}")
        for name, value in metrics.items():
            print(f"  {name:<10}: {value}")

    print(f"\nLatency (ms)")
    print(f"  Retrieval  : {avg_retrieval_ms}")
    print(f"  Total      : {avg_total_ms}")

    print("\n" + "=" * 60)
    print("MLflow UI:")
    print(f"  mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB}")
    print("=" * 60)


if __name__ == "__main__":
    main()