from transformers import pipeline
from retrieve import retrieve
from collections import Counter
import re

print("Loading model...")

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

print("Model loaded.")


# ==========================================================
# Predict labels from retrieved docs (same weighted-vote logic
# as evaluate.py) - giving the LLM the predicted category up
# front often produces a more on-topic, specific response than
# leaving it to infer everything from raw retrieved text alone.
# ==========================================================
def predict_labels(docs, label_keys=("type", "queue", "priority")):
    predictions = {}
    for key in label_keys:
        votes = Counter()
        for doc in docs:
            if key in doc:
                votes[doc[key]] += doc["similarity_score"] ** 2
        predictions[key] = votes.most_common(1)[0][0] if votes else "Unknown"
    return predictions


# ==========================================================
# Clean text
# ==========================================================
def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    # Remove ALL placeholders like <tel_num>, <acc_num>, <email>, etc.
    text = re.sub(r"<[^>]+>", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# Build context from retrieved answers
# Only include answers that are long enough to be useful
# ==========================================================
def build_context(docs):
    context = ""

    for i, doc in enumerate(docs, start=1):
        answer = clean_text(doc.get("answer", ""))

        # Skip empty or very short answers - they add no value
        if len(answer) > 80:
            context += f"Resolution Example {i}: {answer}\n"

    return context.strip()


# ==========================================================
# Generate response
# ==========================================================
def generate_response(query, top_k=5):

    docs = retrieve(query, top_k)

    if not docs:
        return (
            "Sorry, I could not find similar support cases "
            "to help with this issue."
        )

    # ======================================================
    # DEBUG: Print retrieved documents
    # ======================================================
    print("\nRetrieved Documents:\n")

    for i, doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"Document {i}")
        print(f"Similarity : {doc['similarity_score']:.4f}")

        print("\nIssue:")
        print(clean_text(doc["text"][:300]))

        print("\nAnswer:")
        print(clean_text(doc["answer"][:300]))
        print()

    # ======================================================
    # Predict labels from retrieved docs - given to the LLM
    # as upfront context before the retrieved examples
    # ======================================================
    predicted = predict_labels(docs)
    print(f"\nPredicted Type     : {predicted['type']}")
    print(f"Predicted Queue    : {predicted['queue']}")
    print(f"Predicted Priority : {predicted['priority']}")

    # ======================================================
    # Build context from retrieved answers
    # ======================================================
    context = build_context(docs)

    # ======================================================
    # Build prompt as chat messages
    # Qwen2.5-Instruct follows chat format correctly
    # ======================================================
    system_prompt = (
        "You are a professional customer support agent. "
        "Your job is to write a helpful, empathetic, and concise reply "
        "to a customer's support ticket. "
        "Never copy or repeat the customer's own words back to them. "
        "Never include placeholders like <name> or <phone>. "
        "Always be specific and actionable."
    )

    label_context = (
        f"Ticket classification (for your context only - do not state "
        f"these labels directly in your reply):\n"
        f"Type: {predicted['type']}\n"
        f"Queue: {predicted['queue']}\n"
        f"Priority: {predicted['priority']}\n\n"
    )

    if context:
        user_prompt = (
            f"{label_context}"
            f"Customer issue:\n{query}\n\n"
            f"Here are some examples of how similar issues were resolved:\n"
            f"{context}\n\n"
            f"Using the above as inspiration (not copy-paste), write a "
            f"professional support reply in 3 to 5 sentences. "
            f"Acknowledge the issue, apologize briefly, and suggest "
            f"clear next steps."
        )
    else:
        # Fallback if no clean context was found
        user_prompt = (
            f"{label_context}"
            f"Customer issue:\n{query}\n\n"
            f"Write a professional support reply in 3 to 5 sentences. "
            f"Acknowledge the issue, apologize briefly, and suggest "
            f"clear next steps."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    result = generator(
        messages,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.4,
        top_p=0.85,
    )

    # Qwen chat pipeline returns the full message list;
    # the last message is the assistant's reply
    response = result[0]["generated_text"][-1]["content"].strip()

    return response


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":

    query = input("\nEnter support ticket:\n\n")

    response = generate_response(query)

    print("\n" + "=" * 60)
    print("Generated Response")
    print("=" * 60)
    print(response)