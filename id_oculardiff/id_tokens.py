from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPTokenizer


def ensure_1d(x: torch.Tensor) -> torch.Tensor:
    return x.float().view(-1)


def find_id_token_positions(
    tokenizer: CLIPTokenizer,
    prompt: str,
    device: torch.device,
    expected: int = 2,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Tokenize prompt and find positions of placeholder token "id" (expected count = 2).

    We try both " id" and "id" token IDs because CLIP BPE often uses leading space tokens.
    Returns:
      input_ids: [1, max_len]
      positions: list of length 'expected' with token indices in ascending order
    """
    tokenized = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(device)

    candidate_token_ids = []
    for s in [" id", "id"]:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) >= 1:
            candidate_token_ids.append(enc[0])


    positions: List[int] = []
    for tid in candidate_token_ids:
        pos = (input_ids[0] == tid).nonzero(as_tuple=True)[0].tolist()
        for p in pos:
            if p not in positions:
                positions.append(p)

    positions = sorted(positions)
    if len(positions) < expected:
        raise ValueError(
            f"Could not find {expected} 'id' placeholder tokens in prompt.\n"
            f"Prompt: {prompt}\n"
            f"Found positions: {positions}\n"
            f"Try keeping literal 'id id' in the prompt."
        )

    return input_ids, positions[:expected]


def find_id_positions_in_batch_input_ids(
    tokenizer: CLIPTokenizer,
    input_ids: torch.Tensor,
    expected: int = 2,
) -> List[List[int]]:
    """
    For batched tokenized prompts [B, max_len], find positions of placeholder token "id"
    independently for each sample.
    """
    candidate_token_ids = []
    for s in [" id", "id"]:
        enc = tokenizer.encode(s, add_special_tokens=False)
        if len(enc) >= 1:
            candidate_token_ids.append(enc[0])

    positions_per_sample: List[List[int]] = []
    for row in input_ids:
        positions: List[int] = []
        for tid in candidate_token_ids:
            pos = (row == tid).nonzero(as_tuple=True)[0].tolist()
            for p in pos:
                if p not in positions:
                    positions.append(p)

        positions = sorted(positions)
        if len(positions) < expected:
            raise ValueError(
                f"Could not find {expected} 'id' placeholder tokens in tokenized prompt. "
                f"Found positions: {positions}"
            )
        positions_per_sample.append(positions[:expected])

    return positions_per_sample


def inject_id_tokens_no_inplace(
    tok_emb: torch.Tensor,
    positions_per_sample: List[List[int]],
    id_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Replace token embeddings at per-sample `positions_per_sample` with `id_tokens`
    WITHOUT in-place ops.
    """
    bsz, _, dim = tok_emb.shape
    k = id_tokens.shape[1]

    if len(positions_per_sample) != bsz:
        raise ValueError("positions_per_sample length must match batch size")
    if id_tokens.shape != (bsz, k, dim):
        raise ValueError(f"id_tokens must be [B,{k},{dim}]")

    rows = []
    for b in range(bsz):
        positions = positions_per_sample[b]
        if len(positions) != k:
            raise ValueError(f"Sample {b}: expected {k} positions, got {len(positions)}")

        idx = torch.tensor(positions, device=tok_emb.device, dtype=torch.long)
        idx = idx.unsqueeze(-1).expand(-1, dim)


        row = tok_emb[b].scatter(0, idx, id_tokens[b])
        rows.append(row)

    return torch.stack(rows, dim=0)


class IDTokenAdapter(nn.Module):
    """
    Optional small adapter:
    - per-token L2 normalization (recommended)
    - learnable scale to match CLIP token magnitudes
    """
    def __init__(self, dim: int, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.ln = nn.LayerNorm(dim)

    def init_scale_from_token_embedding(self, token_embedding: nn.Embedding):
        with torch.no_grad():
            mean_norm = token_embedding.weight.norm(dim=-1).mean()
            self.scale.copy_(mean_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.normalize:
            x = F.normalize(x, dim=-1)
        x = self.ln(x)
        return x * self.scale
