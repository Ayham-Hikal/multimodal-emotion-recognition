import os

from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def start_training():
    if load_dotenv is not None:
        load_dotenv()

    s3_tensorboard_path = os.getenv("S3_TENSORBOARD_OUTPUT_PATH")
    if not s3_tensorboard_path:
        raise ValueError(
            "Missing S3_TENSORBOARD_OUTPUT_PATH. "
            "Set it in your environment or a .env file."
        )
    s3_train_path = os.getenv("S3_TRAIN_DATA_PATH")
    s3_val_path = os.getenv("S3_VAL_DATA_PATH")
    s3_test_path = os.getenv("S3_TEST_DATA_PATH")
    if not all([s3_train_path, s3_val_path, s3_test_path]):
        raise ValueError(
            "Missing one or more dataset paths: "
            "S3_TRAIN_DATA_PATH, S3_VAL_DATA_PATH, S3_TEST_DATA_PATH. "
            "Set them in your environment or a .env file."
        )

    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path=s3_tensorboard_path,
        container_local_output_path="/opt/ml/output/tensorboard"
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role=os.getenv("AWS_ROLE_ARN"),
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type="ml.g5.xlarge",
        hyperparameters={
            "batch-size": 32,
            "epochs": 25,
            "selection-metric": "emotion_macro_f1",
            "emotion-class-weight-strategy": "none",
            "sentiment-class-weight-strategy": "none",
            "loss-type": "ce",
            "label-smoothing": 0.0,
            "text-unfreeze-layers": 2,
            "head-lr": 5e-4,
            "encoder-lr": 8e-5,
            "text-lr": 8e-6,
        },
        tensorboard_config=tensorboard_config
    )

    # Start training
    estimator.fit({
        "training": s3_train_path,
        "validation": s3_val_path,
        "test": s3_test_path
    })


if __name__ == "__main__":
    start_training()