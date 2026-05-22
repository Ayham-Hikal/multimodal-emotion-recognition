from pathlib import Path

from sagemaker.pytorch import PyTorchModel
import sagemaker
import os


def load_root_env_file():
    """Load key=value pairs from repo-root .env into process env."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    env_path = os.path.join(repo_root, ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def deploy_endpoint():
    load_root_env_file()
    sagemaker.Session()
    role = os.getenv("DEPLOY_ENDPOINT_ROLE") or os.getenv("AWS_ROLE_ARN")

    model_uri = os.getenv("S3_MODEL_URI")
    if not role:
        raise ValueError(
            "Role can not be null. Set DEPLOY_ENDPOINT_ROLE (or AWS_ROLE_ARN) in environment or .env."
        )
    if not model_uri:
        raise ValueError("S3_MODEL_URI is required in environment or .env.")

    backend_root = Path(__file__).resolve().parents[1]

    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        entry_point="deployment/inference.py",
        source_dir=str(backend_root),
        name="sentiment-analysis-model",
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.xlarge",
        endpoint_name="sentiment-analysis-endpoint",
    )


if __name__ == "__main__":
    deploy_endpoint()