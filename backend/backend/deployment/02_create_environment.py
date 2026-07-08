from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment

# =====================================================
# Azure Configuration
# =====================================================

SUBSCRIPTION_ID = "8192d9f0-6361-4953-a574-f9d51f0e407d"
RESOURCE_GROUP = "support-ticket-rg"
WORKSPACE_NAME = "support-ticket-ml"

ENVIRONMENT_NAME = "support-ticket-env"

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
# Create Environment
# =====================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

environment = Environment(
    name=ENVIRONMENT_NAME,
    description="Environment for Support Ticket API",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    conda_file=str(BASE_DIR / "environment.yml"),
)

environment = ml_client.environments.create_or_update(environment)

print("\nEnvironment Created Successfully!")
print("Name    :", environment.name)
print("Version :", environment.version)