import sagemaker
from sagemaker.sklearn.model import SKLearnModel

BUCKET = "asgmd07-aisyahnw07"
MODEL_KEY = "spaceship/model.tar.gz"

ENDPOINT_NAME = "spaceship-endpoint-v4"

ROLE = sagemaker.get_execution_role()

model = SKLearnModel(
    model_data=f"s3://{BUCKET}/{MODEL_KEY}",
    role=ROLE,
    entry_point="inference.py",   
    source_dir="deploy_code",   
    framework_version="1.2-1"
)

print("Sedang mendeploy endpoint v4, mohon tunggu 3-5 menit...")
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=ENDPOINT_NAME
)

print("Deploy selesai!")