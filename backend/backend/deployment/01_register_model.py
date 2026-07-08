from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from pathlib import Path

# =====================================================
# Azure Information
# =====================================================

SUBSCRIPTION_ID = "8192d9f0-6361-4953-a574-f9d51f0e407d"

RESOURCE_GROUP = "support-ticket-rg"

WORKSPACE_NAME = "support-ticket-ml"

MODEL_NAME = "support-ticket-model"

# =====================================================
# Connect
# =====================================================

credential = AzureCliCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

print("Connected to Azure ML Workspace")

# =====================================================
# Register Model
# =====================================================


from pathlib import Path

def register_model():

    print("\nRegistering model...\n")

    # folder that contains deployment.py
    deployment_dir = Path(__file__).resolve().parent

    # deployment/models
    model_dir = deployment_dir / "models"

    print("Deployment Folder :", deployment_dir)
    print("Models Folder     :", model_dir)
    print("Exists            :", model_dir.exists())

    if not model_dir.exists():
        raise FileNotFoundError(f"Models folder not found:\n{model_dir}")

    model = Model(
        path=str(model_dir),
        name=MODEL_NAME,
        description="Support Ticket Classification Models",
        type="custom_model",
    )

    registered_model = ml_client.models.create_or_update(model)

    print("\nModel Registered Successfully")
    print("Name    :", registered_model.name)
    print("Version :", registered_model.version)

    return registered_model

if __name__ == "__main__":
    register_model()