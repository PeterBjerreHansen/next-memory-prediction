from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor


@dataclass(frozen=True)
class MemoryTapeOutput:
    logits_per_pass: tuple[torch.Tensor, ...]
    hidden_states_per_pass: tuple[torch.Tensor, ...]
    memory_states_per_pass: tuple[torch.Tensor, ...]

    @property
    def logits(self) -> torch.Tensor:
        return self.logits_per_pass[-1]

    @property
    def hidden_states(self) -> torch.Tensor:
        return self.hidden_states_per_pass[-1]

    @property
    def memory_states(self) -> torch.Tensor:
        return self.memory_states_per_pass[-1]


class LayerNorm(nn.Module):
    def __init__(self, ndim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, None, 1e-5)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.c_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass(kw_only=True)
class TransformerConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int

    def __post_init__(self):
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.n_layer < 1 or self.n_head < 1 or self.n_embd < 1:
            raise ValueError("model dimensions must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TransformerConfig":
        return cls(**payload)


@dataclass(kw_only=True)
class MemoryTapeConfig(TransformerConfig):
    n_pass: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.n_pass < 2:
            raise ValueError("n_pass must be at least 2")

    @classmethod
    def from_dict(cls, payload: dict) -> "MemoryTapeConfig":
        return cls(**payload)


class CausalTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "ln_f": LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self._init_residual_projections()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_residual_projections(self) -> None:
        std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=std)

    def get_num_params(self, *, non_embedding: bool = True) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            total -= self.transformer["wpe"].weight.numel()
        return total

    def token_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        return self.transformer["wte"](ids)

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block size {self.config.block_size}"
            )
        positions = torch.arange(seq_len, device=ids.device)
        return self.token_embeddings(ids) + self.transformer["wpe"](positions)

    def forward(self, ids: torch.Tensor) -> TransformerOutput:
        x = self._embed_tokens(ids)
        for block in self.transformer["h"]:
            x = block(x)
        hidden_states = self.transformer["ln_f"](x)
        return TransformerOutput(
            logits=self.lm_head(hidden_states),
            hidden_states=hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        do_sample: bool = True,
        inference_mode: str = "recompute",
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if inference_mode != "recompute":
            raise ValueError("CausalTransformer only supports recompute inference")
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -self.config.block_size :]
            logits = self(ids_cond).logits[:, -1, :] / temperature
            if do_sample:
                next_ids = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_ids = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            if eos_token_id is not None and bool((next_ids == eos_token_id).all()):
                break
        return ids


class CausalCrossAttention(nn.Module):
    def __init__(self, config: MemoryTapeConfig):
        super().__init__()
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_kv = nn.Linear(config.n_embd, 2 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        if memory.shape != x.shape:
            raise ValueError("x and memory must have identical shapes")
        q = self.c_q(x)
        k, v = self.c_kv(memory).split(self.n_embd, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.c_proj(y)


class MemoryBlock(nn.Module):
    def __init__(self, config: MemoryTapeConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_mem_q = LayerNorm(config.n_embd)
        self.ln_mem_kv = LayerNorm(config.n_embd)
        self.cross_attn = CausalCrossAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.memory_gate = nn.Parameter(torch.tensor(0.2))

    def memory_gate_scale(self):
        return self.memory_gate

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        memory_delta = self.cross_attn(
            self.ln_mem_q(x),
            self.ln_mem_kv(memory),
        )
        x = x + self.memory_gate_scale() * memory_delta
        x = x + self.mlp(self.ln_2(x))
        return x


class MemoryTapeTransformer(nn.Module):
    """Faithful copy of the upstream MemoryTape core with structured outputs."""

    def __init__(self, config: MemoryTapeConfig):
        super().__init__()
        object.__setattr__(self, "_rng_state_before_construction", torch.get_rng_state())
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([MemoryBlock(config) for _ in range(config.n_layer)]),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "ln_f": LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.ln_mem = LayerNorm(config.n_embd)
        self.mem_head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self._finish_initialization()
        del self._rng_state_before_construction

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_transformer_apply_start_rng_state(self):
        current_state = torch.get_rng_state()
        torch.set_rng_state(self._rng_state_before_construction)
        try:
            nn.ModuleDict(
                {
                    "wte": nn.Embedding(self.config.vocab_size, self.config.n_embd),
                    "h": nn.ModuleList(
                        [
                            TransformerBlock(self.config)
                            for _ in range(self.config.n_layer)
                        ]
                    ),
                    "wpe": nn.Embedding(self.config.block_size, self.config.n_embd),
                    "ln_f": LayerNorm(self.config.n_embd),
                }
            )
            nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
            return torch.get_rng_state()
        finally:
            torch.set_rng_state(current_state)

    def _finish_initialization(self) -> None:
        torch.set_rng_state(self._causal_transformer_apply_start_rng_state())
        self._init_weights(self.transformer["wte"])
        for block in self.transformer["h"]:
            block.attn.apply(self._init_weights)
            block.mlp.apply(self._init_weights)
        self._init_weights(self.transformer["wpe"])
        self._init_weights(self.lm_head)

        residual_std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for block in self.transformer["h"]:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=residual_std)
        for block in self.transformer["h"]:
            block.cross_attn.apply(self._init_weights)
        self._init_weights(self.mem_head)
        for block in self.transformer["h"]:
            nn.init.normal_(
                block.cross_attn.c_proj.weight,
                mean=0.0,
                std=residual_std,
            )

    def get_num_params(self, *, non_embedding: bool = True) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            total -= self.transformer["wpe"].weight.numel()
        return total

    def token_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        return self.transformer["wte"](ids)

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block size {self.config.block_size}"
            )
        positions = torch.arange(seq_len, device=ids.device)
        return self.token_embeddings(ids) + self.transformer["wpe"](positions)

    @staticmethod
    def shift_memory(memory: torch.Tensor) -> torch.Tensor:
        shifted = torch.zeros_like(memory)
        shifted[:, 1:, :] = memory[:, :-1, :]
        return shifted

    def _run_full_pass(
        self,
        token_stream: torch.Tensor,
        memory_tape: torch.Tensor,
    ) -> torch.Tensor:
        x = token_stream
        for block in self.transformer["h"]:
            x = block(x, memory_tape)
        return x

    def forward_passes(self, ids: torch.Tensor) -> MemoryTapeOutput:
        token_stream = self._embed_tokens(ids)
        previous_memory = torch.zeros_like(token_stream)
        logits_per_pass = []
        hidden_per_pass = []
        memory_per_pass = []

        for _ in range(self.config.n_pass):
            raw_hidden = self._run_full_pass(
                token_stream,
                self.shift_memory(previous_memory),
            )
            hidden = self.transformer["ln_f"](raw_hidden)
            logits = self.lm_head(hidden)
            memory = self.mem_head(self.ln_mem(raw_hidden))
            logits_per_pass.append(logits)
            hidden_per_pass.append(hidden)
            memory_per_pass.append(memory)
            previous_memory = memory

        return MemoryTapeOutput(
            logits_per_pass=tuple(logits_per_pass),
            hidden_states_per_pass=tuple(hidden_per_pass),
            memory_states_per_pass=tuple(memory_per_pass),
        )

    def forward(self, ids: torch.Tensor) -> MemoryTapeOutput:
        return self.forward_passes(ids)

    def memory_gate_stats(self) -> dict[str, object] | None:
        values = []
        for block in self.transformer["h"]:
            raw = float(block.memory_gate.detach().cpu())
            effective = float(
                torch.as_tensor(block.memory_gate_scale()).detach().cpu()
            )
            values.append((raw, effective))
        effective_values = [item[1] for item in values]
        return {
            "mode": "scalar",
            "raw": [item[0] for item in values],
            "effective": effective_values,
            "mean_abs_effective": sum(map(abs, effective_values)) / len(values),
            "max_abs_effective": max(map(abs, effective_values)),
        }

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        do_sample: bool = True,
        inference_mode: str = "recompute",
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if inference_mode == "recompute":
            for _ in range(max_new_tokens):
                ids_cond = ids[:, -self.config.block_size :]
                logits = self(ids_cond).logits[:, -1, :] / temperature
                next_ids = (
                    torch.multinomial(F.softmax(logits, dim=-1), 1)
                    if do_sample
                    else logits.argmax(dim=-1, keepdim=True)
                )
                ids = torch.cat((ids, next_ids), dim=1)
                if eos_token_id is not None and bool((next_ids == eos_token_id).all()):
                    break
            return ids
        if inference_mode != "final_pass":
            raise ValueError("inference_mode must be recompute or final_pass")
        return self._generate_final_pass(
            ids,
            max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
        )

    def _generate_final_pass(
        self,
        ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float,
        do_sample: bool,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            return ids

        ids_window = ids[:, -self.config.block_size :]
        output = self(ids_window)
        logits = output.logits
        memory_history = output.memory_states

        for generated_index in range(max_new_tokens):
            next_logits = logits[:, -1, :] / temperature
            next_ids = (
                torch.multinomial(F.softmax(next_logits, dim=-1), 1)
                if do_sample
                else next_logits.argmax(dim=-1, keepdim=True)
            )
            ids = torch.cat((ids, next_ids), dim=1)
            if eos_token_id is not None and bool((next_ids == eos_token_id).all()):
                break
            if generated_index + 1 == max_new_tokens:
                break

            ids_window = ids[:, -self.config.block_size :]
            token_stream = self._embed_tokens(ids_window)
            memory_tape = torch.zeros_like(token_stream)
            if token_stream.size(1) > 1:
                history = memory_history[:, -(token_stream.size(1) - 1) :, :]
                memory_tape[:, 1:, :] = history
            raw_hidden = self._run_full_pass(token_stream, memory_tape)
            hidden = self.transformer["ln_f"](raw_hidden)
            logits = self.lm_head(hidden)
            current_memory = self.mem_head(self.ln_mem(raw_hidden))
            memory_history = torch.cat(
                (memory_history, current_memory[:, -1:, :]),
                dim=1,
            )[:, -self.config.block_size :, :]
        return ids


def round_to_multiple(value: float, multiple: int) -> int:
    return multiple * round(value / multiple)


class LatentTransitionPredictor(nn.Module):
    """Training-only horizon-one residual latent-transition model."""

    def __init__(self, n_embd: int, *, projection_factor: float = 1.3):
        super().__init__()
        input_dim = 2 * n_embd
        hidden_dim = max(128, round_to_multiple(input_dim * projection_factor, 128))
        self.norm = LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd, bias=False),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        current_latent: torch.Tensor,
        next_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        inputs = self.norm(
            torch.cat((next_token_embeddings, current_latent), dim=-1)
        )
        return current_latent + self.mlp(inputs)

    def get_num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
