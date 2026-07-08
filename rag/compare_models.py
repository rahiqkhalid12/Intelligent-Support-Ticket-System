import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Paths
# ==========================================================

TRADITIONAL_PATH = "reports/traditional/summary.json"

BERT_DIR = "reports/bert"

RAG_PATH = "reports/rag/evaluation_results.txt"

OUTPUT_DIR = "reports/comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tasks = ["type", "queue", "priority"]

# ==========================================================
# Traditional
# ==========================================================

with open(TRADITIONAL_PATH, "r") as f:
    traditional = json.load(f)

trad_acc = []
trad_prec = []
trad_rec = []
trad_f1 = []

for task in tasks:

    trad_acc.append(
        traditional[task]["accuracy"] * 100
    )

    trad_prec.append(
        traditional[task]["precision"] * 100
    )

    trad_rec.append(
        traditional[task]["recall"] * 100
    )

    trad_f1.append(
        traditional[task]["f1_score"] * 100
    )

# ==========================================================
# DistilBERT
# ==========================================================

bert_acc = []
bert_prec = []
bert_rec = []
bert_f1 = []

for task in tasks:

    with open(f"{BERT_DIR}/{task}.json", "r") as f:

        data = json.load(f)

    bert_acc.append(
        data["accuracy"] * 100
    )

    bert_prec.append(
        data["precision"] * 100
    )

    bert_rec.append(
        data["recall"] * 100
    )

    bert_f1.append(
        data["f1_score"] * 100
    )

# ==========================================================
# RAG
# ==========================================================

with open(RAG_PATH, "r", encoding="utf-8") as f:

    text = f.read()

rag_acc = []
rag_prec = []
rag_rec = []
rag_f1 = []

sections = ["TYPE", "QUEUE", "PRIORITY"]

for section in sections:

    pattern = rf"{section}.*?Accuracy:\s*([\d\.]+)(.*?)(?=\n=+|\Z)"

    match = re.search(
        pattern,
        text,
        re.DOTALL
    )

    if match:

        accuracy = float(match.group(1))

        block = match.group(2)

        weighted = re.search(

            r"weighted avg\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)",

            block

        )

        precision = float(weighted.group(1))

        recall = float(weighted.group(2))

        f1 = float(weighted.group(3))

        rag_acc.append(
            accuracy * 100
        )

        rag_prec.append(
            precision * 100
        )

        rag_rec.append(
            recall * 100
        )

        rag_f1.append(
            f1 * 100
        )

# ==========================================================
# BLEU
# ==========================================================

bleu = re.search(

    r"Average BLEU:\s*([\d\.]+)",

    text

)

bleu_score = float(bleu.group(1))

# ==========================================================
# DataFrame
# ==========================================================

df = pd.DataFrame({

    "Task": ["Type", "Queue", "Priority"],

    "Traditional_Accuracy": trad_acc,
    "DistilBERT_Accuracy": bert_acc,
    "RAG_Accuracy": rag_acc,

    "Traditional_Precision": trad_prec,
    "DistilBERT_Precision": bert_prec,
    "RAG_Precision": rag_prec,

    "Traditional_Recall": trad_rec,
    "DistilBERT_Recall": bert_rec,
    "RAG_Recall": rag_rec,

    "Traditional_F1": trad_f1,
    "DistilBERT_F1": bert_f1,
    "RAG_F1": rag_f1,

})

# ==========================================================
# Generic Plot Function
# ==========================================================

def create_bar_plot(metric):

    x = np.arange(3)

    width = 0.25

    plt.figure(figsize=(10,6))

    plt.bar(

        x-width,

        df[f"Traditional_{metric}"],

        width,

        label="Traditional"

    )

    plt.bar(

        x,

        df[f"DistilBERT_{metric}"],

        width,

        label="DistilBERT"

    )

    plt.bar(

        x+width,

        df[f"RAG_{metric}"],

        width,

        label="RAG"

    )

    plt.xticks(

        x,

        ["Type","Queue","Priority"]

    )

    plt.ylabel(f"{metric} (%)")

    plt.title(f"{metric} Comparison")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        f"{OUTPUT_DIR}/{metric.lower()}_comparison.png"

    )

    plt.close()

# ==========================================================
# Generate 4 bar plots
# ==========================================================

for metric in [

    "Accuracy",

    "Precision",

    "Recall",

    "F1"

]:

    create_bar_plot(metric)

# ==========================================================
# Radar Chart
# ==========================================================

categories = [

    "Type",

    "Queue",

    "Priority"

]

N = len(categories)

angles = np.linspace(

    0,

    2*np.pi,

    N,

    endpoint=False

)

angles = np.concatenate(

    (angles,[angles[0]])

)

trad = trad_acc + [trad_acc[0]]

bert = bert_acc + [bert_acc[0]]

rag = rag_acc + [rag_acc[0]]

fig = plt.figure(

    figsize=(8,8)

)

ax = fig.add_subplot(

    111,

    polar=True

)

ax.plot(

    angles,

    trad,

    linewidth=2,

    label="Traditional"

)

ax.fill(

    angles,

    trad,

    alpha=0.1

)

ax.plot(

    angles,

    bert,

    linewidth=2,

    label="DistilBERT"

)

ax.fill(

    angles,

    bert,

    alpha=0.1

)

ax.plot(

    angles,

    rag,

    linewidth=2,

    label="RAG"

)

ax.fill(

    angles,

    rag,

    alpha=0.1

)

ax.set_xticks(

    angles[:-1]

)

ax.set_xticklabels(

    categories

)

plt.title(

    "Overall Performance"

)

plt.legend()

plt.savefig(

    f"{OUTPUT_DIR}/radar_chart.png"

)

plt.close()

# ==========================================================
# Heatmap
# ==========================================================

heatmap_data = pd.DataFrame({

    "Traditional": trad_acc,

    "DistilBERT": bert_acc,

    "RAG": rag_acc,

},

index=["Type","Queue","Priority"])

plt.figure(figsize=(8,6))

plt.imshow(

    heatmap_data,

    aspect="auto"

)

plt.xticks(

    range(3),

    heatmap_data.columns

)

plt.yticks(

    range(3),

    heatmap_data.index

)

plt.colorbar()

plt.title(

    "Accuracy Heatmap"

)

plt.tight_layout()

plt.savefig(

    f"{OUTPUT_DIR}/heatmap.png"

)

plt.close()

# ==========================================================
# BLEU Plot
# ==========================================================

plt.figure(figsize=(6,5))

plt.bar(

    ["RAG"],

    [bleu_score]

)

plt.ylabel(

    "BLEU Score"

)

plt.title(

    "RAG Response Quality"

)

plt.tight_layout()

plt.savefig(

    f"{OUTPUT_DIR}/bleu_score.png"

)

plt.close()

print("\nDone.")

print(f"Saved to {OUTPUT_DIR}")