from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
)

SUBSCRIPTION_ID = "YOUR SUBSCRIPTION_ID"
RESOURCE_GROUP  = "YOUR RESOURCE_GROUP "
WORKSPACE_NAME  = "YOUR WORKSPACE_NAME"
ENDPOINT_NAME   = "YOUR ENDPOINT_NAME"

credential = AzureCliCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

print("Connected to workspace.")

# ── 1. Create Endpoint ──────────────────────────────────
print("\n[1/4] Creating endpoint...")

endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    auth_mode="key",
    description="Support Ticket Classification API",
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Endpoint ready.")

# ── 2. Create Deployment ────────────────────────────────
print("\n[2/4] Creating deployment (this takes 5-10 minutes)...")

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=ENDPOINT_NAME,
    model="azureml:support-ticket-model:1",
    environment="azureml:support-ticket-env:1",
    code_configuration=CodeConfiguration(
        code=".",
        scoring_script="score.py",
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(deployment).result()
print("Deployment ready.")

# ── 3. Route 100% traffic to blue ──────────────────────
print("\n[3/4] Setting traffic...")

endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Traffic set to 100% → blue")

# ── 4. Print URL and Key ────────────────────────────────
print("\n[4/4] Getting endpoint details...")

endpoint_info = ml_client.online_endpoints.get(ENDPOINT_NAME)
keys = ml_client.online_endpoints.get_keys(ENDPOINT_NAME)

print("\n" + "=" * 55)
print("  ENDPOINT URL :", endpoint_info.scoring_uri)
print("  PRIMARY KEY  :", keys.primary_key)
print("=" * 55)
print("\nDeployment complete! Save the URL and KEY above.")