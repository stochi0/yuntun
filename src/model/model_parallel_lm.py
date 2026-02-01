"""
Model-parallel aware causal language model interface.

This module is responsible for:
- Loading sharded / non-sharded models
- Exposing unified forward / generate APIs
- Integrating with tensor + pipeline parallel backends

Similar role to GPTModel / MegatronModule in Megatron-LM.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelParallelCausalLM:
    """
    High-level interface for a causal LM running in a
    tensor/pipeline/data parallel environment.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code

        self.model = None
        self.tokenizer = None

    # ----------------------------------------------------
    # Initialization
    # ----------------------------------------------------

    def initialize(self):
        """
        Load model + tokenizer.

        Later:
        - Replace with sharded checkpoint loader
        - Register tensor/pipeline parallel layers
        """

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=self.trust_remote_code
        ).to(self.device)

        self.model.eval()

    # ----------------------------------------------------
    # Core interfaces
    # ----------------------------------------------------

    def forward(self, input_ids, attention_mask=None):

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        **generation_kwargs
    ):
        """
        Autoregressive generation.

        In model-parallel mode this will be overridden
        with distributed sampling.
        """

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs
        )

    # ----------------------------------------------------
    # Accessors
    # ----------------------------------------------------

    def parameters(self):
        return self.model.parameters()

    def get_tokenizer(self):
        return self.tokenizer

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    # ----------------------------------------------------
    # Checkpoint hooks (future)
    # ----------------------------------------------------

    def save_sharded_checkpoint(self, path: str):
        """
        Placeholder for Megatron-style sharded saving.
        """
        raise NotImplementedError

    def load_sharded_checkpoint(self, path: str):
        """
        Placeholder for Megatron-style sharded loading.
        """
        raise NotImplementedError
