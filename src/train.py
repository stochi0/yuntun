from pathlib import Path
import torch
from config import load_config
from models.qwen3 import Qwen3Model, Qwen3ModelConfig

def train():
    # Load training config
    # Assuming running from src/ or root, adjust path accordingly
    # meaningful path relative to script location
    config_path = Path(__file__).parents[1] / "configs" / "dev.json"
    train_cfg = load_config(str(config_path))
    print(f"Training with config: {train_cfg}")

    # Initialize model
    model_cfg = Qwen3ModelConfig()
    model = Qwen3Model(model_cfg)
    print("Model initialized")

    # Dummy train loop...
    pass

if __name__ == "__main__":
    train()
