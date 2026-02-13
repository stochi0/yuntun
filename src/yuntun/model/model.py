import torch
import torch.nn as nn
from typing import Optional, List

from .config import Qwen3Config
from .layers import Qwen3Block, RMSNorm
from .utils import compute_rope_params

# Attempt tp imports
try:
    from ..distributed.tensor_parallel import (
        VocabParallelEmbedding,
        ParallelLMHead,
    )

    TP_AVAILABLE = True
except ImportError:
    TP_AVAILABLE = False
    VocabParallelEmbedding = None
    ParallelLMHead = None


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config, tp_group=None):
        super().__init__()
        self.config = config
        self.tp_group = tp_group

        self.embed_tokens = self._init_embeddings(
            config.vocab_size, config.hidden_size, tp_group
        )

        self.layers = nn.ModuleList(
            [
                Qwen3Block(config, layer_idx=i, tp_group=tp_group)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize RoPE cache
        self.head_dim = config.hidden_size // config.num_attention_heads
        cos, sin = compute_rope_params(
            self.head_dim,
            config.rope_theta,
            config.max_position_embeddings,
            device=torch.device("cpu"),  # Compute on CPU initially or let it be moved
            dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _init_embeddings(self, vocab_size, hidden_size, tp_group):
        if tp_group is not None and tp_group.size() > 1 and TP_AVAILABLE:
            return VocabParallelEmbedding(vocab_size, hidden_size, tp_group=tp_group)
        else:
            return nn.Embedding(vocab_size, hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Prepare RoPE (move to device if needed, usually buffer handles it)
        cos = self.cos
        sin = self.sin

        if cos.device != hidden_states.device:
            cos = cos.to(hidden_states.device)
            sin = sin.to(hidden_states.device)

        next_decoder_cache = [] if use_cache else None

        for i, layer_module in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None

            hidden_states, present_kv = layer_module(
                hidden_states,
                cos,
                sin,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
            )

            if next_decoder_cache is not None:
                next_decoder_cache.append(present_kv)

        hidden_states = self.norm(hidden_states)

        return hidden_states, next_decoder_cache


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config, tp_group=None):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, tp_group=tp_group)

        if tp_group is not None and tp_group.size() > 1 and TP_AVAILABLE:
            self.lm_head = ParallelLMHead(
                config.hidden_size, config.vocab_size, bias=False, tp_group=tp_group
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if needed (Qwen usually doesn't, but standard practice check)
        # if config.tie_word_embeddings:
        #     self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs[1],
            "hidden_states": outputs[0],  # Approximate
        }
