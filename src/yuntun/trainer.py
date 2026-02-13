"""Training loop for Qwen3-style causal LM on FineWeb."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .model.config import Qwen3Config
from .model.model import Qwen3ForCausalLM
from .data.fineweb import FineWebDataModule
from .distributed.parallel import create_groups


def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup then linear decay."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Simple trainer for causal LM pre-training."""

    def __init__(
        self,
        config_path: str,
        device: str | None = None,
    ):
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.tp_size = self.cfg.get("tp_size", 1)
        self.tp_group = None

        if self.tp_size > 1:
            if not dist.is_available():
                raise RuntimeError("Tensor parallelism requires torch.distributed")
            if not dist.is_initialized():
                if "RANK" not in os.environ:
                    raise RuntimeError(
                        "tp_size > 1 requires torchrun. Run: "
                        "torchrun --nproc_per_node=N scripts/train.py --config <path>"
                    )
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo"
                )
            world_size = dist.get_world_size()
            if world_size != self.tp_size:
                raise ValueError(
                    f"tp_size={self.tp_size} must match world_size={world_size}. "
                    f"Use: torchrun --nproc_per_node={self.tp_size} ..."
                )
            groups = create_groups(tp_size=self.tp_size, pp_size=1, dp_size=1)
            self.tp_group = groups.tp

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.tp_size > 1 and torch.cuda.is_available():
            rank = dist.get_rank()
            self.device = f"cuda:{rank}"

        self.output_dir = Path(self.cfg["output_dir"])
        if not dist.is_initialized() or dist.get_rank() == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build model config
        model_cfg = self.cfg.get("model", {})
        self.model_config = Qwen3Config(
            vocab_size=model_cfg.get("vocab_size", 151936),
            hidden_size=model_cfg.get("hidden_size", 1024),
            intermediate_size=model_cfg.get("intermediate_size", 3072),
            num_hidden_layers=model_cfg.get("num_hidden_layers", 28),
            num_attention_heads=model_cfg.get("num_attention_heads", 16),
            num_key_value_heads=model_cfg.get("num_key_value_heads", 8),
            max_position_embeddings=model_cfg.get("max_position_embeddings", 40960),
        )

        self.model = Qwen3ForCausalLM(self.model_config, tp_group=self.tp_group).to(
            self.device
        )
        self.data_module = FineWebDataModule(
            tokenizer_name=self.cfg.get("tokenizer", "Qwen/Qwen3-0.6B"),
            max_length=self.cfg.get("max_seq_length", 512),
            batch_size=self.cfg.get("batch_size", 8),
            seed=42,
        )

        self.batch_size = self.cfg["batch_size"]
        self.grad_accum_steps = self.cfg.get("grad_accum_steps", 1)
        self.max_steps = self.cfg["max_steps"]
        self.eval_steps = self.cfg.get("eval_steps", 500)
        self.save_steps = self.cfg.get("save_steps", 1000)
        self.warmup_steps = self.cfg.get("warmup_steps", 500)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.get("lr", 3e-4),
            weight_decay=self.cfg.get("weight_decay", 0.01),
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.max_steps,
        )

        self.global_step = 0
        self.accum_loss = 0.0
        self.accum_count = 0

    def train(self):
        """Run training loop."""
        self.model.train()
        dataloader = self.data_module.train_dataloader()
        dl_iter = iter(dataloader)

        while self.global_step < self.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            local_loss = 0.0

            for _ in range(self.grad_accum_steps):
                try:
                    batch = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(dataloader)
                    batch = next(dl_iter)

                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    return_dict=True,
                )
                loss = outputs["loss"] / self.grad_accum_steps
                loss.backward()
                local_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1
            self.accum_loss += local_loss
            self.accum_count += 1

            if self.global_step % 10 == 0 or self.global_step <= 5:
                avg = self.accum_loss / self.accum_count
                print(f"Step {self.global_step}/{self.max_steps}  loss={avg:.4f}")
                self.accum_loss = 0.0
                self.accum_count = 0

            if self.global_step % self.eval_steps == 0 and self.global_step > 0:
                self._log_and_eval()

            if self.global_step % self.save_steps == 0 and self.global_step > 0:
                self.save_checkpoint()

        self.save_checkpoint()
        print("Training complete.")

    def _log_and_eval(self):
        """Optional: quick eval pass or just log."""
        self.model.eval()
        with torch.no_grad():
            # Placeholder: could run a few batches for eval loss
            pass
        self.model.train()

    def save_checkpoint(self):
        """Save model and optimizer state."""
        ckpt_path = self.output_dir / f"checkpoint-{self.global_step}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if self.tp_size > 1 and dist.is_initialized():
            suffix = f"_rank{dist.get_rank()}"
        else:
            suffix = ""
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.global_step,
                "config": self.cfg,
            },
            ckpt_path / f"pytorch_model{suffix}.pt",
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Saved checkpoint to {ckpt_path}")
