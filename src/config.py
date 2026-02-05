from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class TrainingConfig:
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    K: int = 8
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    max_gen_len: int = 2048

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        with open(path, "r") as f:
            data = json.load(f)
        # Only keep keys that are fields in the dataclass
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

def load_config(config_path: str) -> TrainingConfig:
    return TrainingConfig.from_json(config_path)
