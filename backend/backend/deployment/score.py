import json
import os
import traceback
import joblib

# ======================================================
# Global Variables
# ======================================================

vectorizer = None

priority_model = None
queue_model = None
type_model = None

priority_mapping = None
queue_mapping = None
type_mapping = None


# ======================================================
# Initialization
# ======================================================

def init():
    print("Initializing Azure ML Endpoint...")
    """
    Runs once when the endpoint starts.
    Loads all required models into memory.
    """

    global vectorizer
    global priority_model
    global queue_model
    global type_model

    global priority_mapping
    global queue_mapping
    global type_mapping

    # Azure Deployment
    model_path = os.getenv("AZUREML_MODEL_DIR")

    # Local Testing
    if model_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "models")

    print("=" * 60)
    print("Model Path:", model_path)
    print("=" * 60)

    print("=" * 60)
    print("Loading Support Ticket Models...")
    print("=" * 60)

    # Load Vectorizer
    vectorizer = joblib.load(
        os.path.join(model_path, "tfidf_vectorizer.joblib")
    )

    # Load Models
    priority_model = joblib.load(
        os.path.join(model_path, "priority_best_model.joblib")
    )

    queue_model = joblib.load(
        os.path.join(model_path, "queue_best_model.joblib")
    )

    type_model = joblib.load(
        os.path.join(model_path, "type_best_model.joblib")
    )

    # Load Label Mapping
    with open(
        os.path.join(model_path, "label_mappings.json"),
        "r",
        encoding="utf-8",
    ) as f:

        mappings = json.load(f)

    priority_mapping = {
        int(v): k
        for k, v in mappings["priority"].items()
    }

    queue_mapping = {
        int(v): k
        for k, v in mappings["queue"].items()
    }

    type_mapping = {
        int(v): k
        for k, v in mappings["type"].items()
    }

    print("Models Loaded Successfully.")
    print("=" * 60)


# ======================================================
# Prediction Function
# ======================================================

def run(raw_data):
    """
    Azure calls this function for every request.
    """

    try:

        # Parse Input
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        text = data.get("text", "").strip()

        if len(text) == 0:
            return {
                "status": "error",
                "message": "Input text is empty."
            }

        # Vectorize
        features = vectorizer.transform([text])

        # Predict
        priority_pred = int(priority_model.predict(features)[0])
        queue_pred = int(queue_model.predict(features)[0])
        type_pred = int(type_model.predict(features)[0])

        # Decode Labels
        priority = priority_mapping.get(priority_pred, "Unknown")
        queue = queue_mapping.get(queue_pred, "Unknown")
        ticket_type = type_mapping.get(type_pred, "Unknown")

        return {
            "status": "success",
            "prediction": {
                "priority": priority,
                "queue": queue,
                "type": ticket_type
            }
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }