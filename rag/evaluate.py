import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from retrieve import retrieve
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

TOP_K = 10              # candidate pool size - larger than before so the
                         # threshold below has good matches to choose from
SIMILARITY_THRESHOLD = 0.55  # discard weak/noisy matches before voting
MIN_VOTES = 1            # always keep at least this many docs even if all
                          # are below threshold, so we never vote on nothing


def filter_by_similarity(docs):
    """Drop retrieved tickets below SIMILARITY_THRESHOLD - a loosely
    related ticket (e.g. similarity 0.40) adds noise to the vote
    rather than signal. Falls back to the single best match if
    nothing clears the threshold, so we're never voting on an empty
    set."""
    strong = [d for d in docs if d["similarity_score"] >= SIMILARITY_THRESHOLD]
    if len(strong) >= MIN_VOTES:
        return strong
    return docs[:MIN_VOTES] if docs else []


def weighted_vote(docs, label_key):
    """Predict a label by similarity-weighted vote across filtered
    retrieved tickets, instead of a single 1-NN lookup or unweighted
    majority vote.

    Each retrieved ticket "votes" for its own label, weighted by its
    SQUARED similarity score - this sharpens the gap between close
    and loose matches more than linear weighting does (e.g. 0.9 vs
    0.6 becomes 0.81 vs 0.36, a bigger relative gap than 0.9 vs 0.6
    directly), so a few very close matches dominate the vote instead
    of being diluted by several mediocre ones."""
    docs = filter_by_similarity(docs)
    votes = Counter()
    for doc in docs:
        votes[doc[label_key]] += doc["similarity_score"] ** 2
    return votes.most_common(1)[0][0]


def best_answer(docs):
    """For BLEU/response comparison, use the single most similar
    ticket's answer (voting doesn't apply to free-text generation,
    so this stays nearest-neighbor by design)."""
    return docs[0].get("answer", "") if docs else ""



# ==========================================================
# Load test data
# ==========================================================
TEST_PATH = "data/processed/test.csv"

test_df = pd.read_csv(TEST_PATH)

true_type = []
pred_type = []

true_queue = []
pred_queue = []

true_priority = []
pred_priority = []

bleu_scores = []

smooth = SmoothingFunction().method1

# ==========================================================
# Evaluate
# ==========================================================
for _, row in test_df.iterrows():

    query = str(row["text"])

    docs = retrieve(query, top_k=TOP_K)

    if len(docs) == 0:
        continue

    # ------------------------
    # Classification: similarity-weighted majority vote across
    # top-5 retrieved tickets (not just the single nearest neighbor)
    # ------------------------
    true_type.append(row["type"])
    pred_type.append(weighted_vote(docs, "type"))

    true_queue.append(row["queue"])
    pred_queue.append(weighted_vote(docs, "queue"))

    true_priority.append(row["priority"])
    pred_priority.append(weighted_vote(docs, "priority"))

    # ------------------------
    # BLEU for responses (uses nearest-neighbor's answer - voting
    # doesn't apply to free text the same way it does to labels)
    # ------------------------
    reference = str(row.get("answer", "")).split()

    candidate = best_answer(docs).split()

    if len(reference) > 0 and len(candidate) > 0:
        bleu = sentence_bleu(
            [reference],
            candidate,
            smoothing_function=smooth,
        )
        bleu_scores.append(bleu)

# ==========================================================
# Confusion Matrices
# ==========================================================

# Type
cm = confusion_matrix(true_type, pred_type)
disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(true_type)))
disp.plot(xticks_rotation=45)
plt.title("RAG - Type Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/rag/type_confusion_matrix.png")
plt.close()

# Queue
cm = confusion_matrix(true_queue, pred_queue)
disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(true_queue)))
disp.plot(xticks_rotation=45)
plt.title("RAG - Queue Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/rag/queue_confusion_matrix.png")
plt.close()

# Priority
cm = confusion_matrix(true_priority, pred_priority)
disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(true_priority)))
disp.plot(xticks_rotation=45)
plt.title("RAG - Priority Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/rag/priority_confusion_matrix.png")
plt.close()

print("Confusion matrices saved.")

# ==========================================================
# Results
# ==========================================================

output_file = "evaluation_results.txt"

with open(output_file, "w", encoding="utf-8") as f:

    # TYPE
    f.write("\n" + "=" * 60 + "\n")
    f.write("TYPE\n")
    f.write("=" * 60 + "\n")

    type_acc = accuracy_score(true_type, pred_type)
    f.write(f"Accuracy: {type_acc:.4f}\n")
    f.write(classification_report(true_type, pred_type, digits=4))
    f.write("\n")

    # QUEUE
    f.write("\n" + "=" * 60 + "\n")
    f.write("QUEUE\n")
    f.write("=" * 60 + "\n")

    queue_acc = accuracy_score(true_queue, pred_queue)
    f.write(f"Accuracy: {queue_acc:.4f}\n")
    f.write(classification_report(true_queue, pred_queue, digits=4))
    f.write("\n")

    # PRIORITY
    f.write("\n" + "=" * 60 + "\n")
    f.write("PRIORITY\n")
    f.write("=" * 60 + "\n")

    priority_acc = accuracy_score(true_priority, pred_priority)
    f.write(f"Accuracy: {priority_acc:.4f}\n")
    f.write(classification_report(true_priority, pred_priority, digits=4))
    f.write("\n")

    # BLEU
    f.write("\n" + "=" * 60 + "\n")
    f.write("BLEU SCORE\n")
    f.write("=" * 60 + "\n")

    if len(bleu_scores) > 0:
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        f.write(f"Average BLEU: {avg_bleu:.4f}\n")
    else:
        f.write("No BLEU score could be computed.\n")

# Also print to the terminal
print(f"\nEvaluation results saved to: {output_file}")