"""FineWeb dataset loading and tokenization for causal LM training."""

from typing import Optional
import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

FINEWEB_SAMPLE = "sample-10BT"  # 14.9M rows, ~10B tokens
FINEWEB_DATASET = "HuggingFaceFW/fineweb"


class FineWebIterableDataset(IterableDataset):
    """Iterable dataset that streams FineWeb and tokenizes on the fly."""

    def __init__(
        self,
        split: str = "train",
        tokenizer=None,
        max_length: int = 512,
        split_text: bool = True,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split_text = split_text
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def __iter__(self):
        from datasets import load_dataset

        dataset = load_dataset(
            FINEWEB_DATASET,
            name=FINEWEB_SAMPLE,
            split=self.split,
            streaming=True,
        )

        buffer = []
        rng = torch.Generator().manual_seed(self.seed)

        for example in dataset:
            text = example.get("text", "")
            if not text or not text.strip():
                continue

            if self.split_text:
                # Split long docs into chunks of max_length tokens
                ids = self.tokenizer(
                    text,
                    truncation=False,
                    add_special_tokens=True,
                    return_attention_mask=False,
                )["input_ids"]
                for i in range(0, len(ids), self.max_length - 1):
                    chunk = ids[i : i + self.max_length]
                    if len(chunk) >= 32:  # Skip very short chunks
                        buffer.append(chunk)
            else:
                ids = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=True,
                    return_attention_mask=False,
                )["input_ids"]
                if len(ids) >= 32:
                    buffer.append(ids)

            # Yield batches from buffer with shuffling
            while len(buffer) >= self.shuffle_buffer_size:
                if self.shuffle_buffer_size > 1:
                    perm = torch.randperm(len(buffer), generator=rng).tolist()
                    buffer = [buffer[i] for i in perm]
                for _ in range(min(self.shuffle_buffer_size, len(buffer))):
                    chunk = buffer.pop(0)
                    yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}

        # Yield remaining
        for chunk in buffer:
            yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


def load_fineweb_dataset(
    tokenizer=None,
    max_length: int = 512,
    split: str = "train",
    split_text: bool = True,
    shuffle_buffer_size: int = 10_000,
    seed: int = 42,
) -> FineWebIterableDataset:
    """Load FineWeb as an iterable dataset for training."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return FineWebIterableDataset(
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        split_text=split_text,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
    )


def collate_fn(batch, pad_token_id: int = 0):
    """Pad sequences to same length in batch. Labels use -100 for padding (no loss)."""
    input_ids = [b["input_ids"] for b in batch]
    max_len = max(x.size(0) for x in input_ids)
    padded_ids = torch.stack(
        [
            torch.nn.functional.pad(x, (0, max_len - x.size(0)), value=pad_token_id)
            for x in input_ids
        ]
    )
    labels = padded_ids.clone()
    labels[labels == pad_token_id] = -100  # Ignore padding in loss
    return {"input_ids": padded_ids, "labels": labels}


class FineWebDataModule:
    """Data module for FineWeb training."""

    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
        max_length: int = 512,
        batch_size: int = 8,
        split_text: bool = True,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
        num_workers: int = 0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.split_text = split_text
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = load_fineweb_dataset(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            split="train",
            split_text=self.split_text,
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=self.seed,
        )
        pad_id = self.tokenizer.pad_token_id or 0
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=lambda b: collate_fn(b, pad_token_id=pad_id),
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
