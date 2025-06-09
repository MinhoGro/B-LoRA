#!/usr/bin/env python
# coding=utf-8
"""End-to-end Style-LoRA × Prompt-OT training script.

This script demonstrates a minimal pipeline that follows the
"Style-LoRA × Prompt-OT" scheme described in the repository README.
It includes:
 1. Prompt decomposition via Optimal Transport (Prompt-OT).
 2. Sinkhorn based OT-Attention with learnable cost projections.
 3. Two-stage training: OT distillation followed by one-shot LoRA fine-tuning.
 4. Inference helper loading the trained LoRA and OT parameters.

The implementation is intentionally compact and serves as a reference for
research experiments. It relies on `diffusers` and `peft` for the
underlying Stable Diffusion XL components.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer


# -----------------------------------------------------------------------------
# Prompt OT
# -----------------------------------------------------------------------------
@dataclass
class PromptOTResult:
    style_prompt: str
    content_prompt: str
    gamma: torch.Tensor


def prompt_ot_split(
    prompt: str,
    tokenizer: AutoTokenizer,
    text_encoder: nn.Module,
    eps: float = 0.05,
    sinkhorn_iters: int = 5,
    threshold: float = 0.1,
) -> PromptOTResult:
    """Split a prompt into style/content parts using OT."""
    inputs = tokenizer(prompt, return_tensors="pt")
    tokens = inputs.input_ids.to(text_encoder.device)
    embeds = text_encoder.get_input_embeddings()(tokens)

    # K=2 anchors for style/content (learnable during training)
    z = nn.Parameter(torch.randn(2, embeds.size(-1), device=embeds.device))

    cost = (embeds.pow(2).sum(-1, keepdim=True) + z.pow(2).sum(-1) - 2 * embeds @ z.T)
    log_scores = -cost / eps
    u = torch.full_like(log_scores[..., 0], 1.0 / log_scores.size(-1))
    for _ in range(sinkhorn_iters):
        v = 1.0 / (log_scores.exp().transpose(-1, -2) @ u.unsqueeze(-1)).squeeze(-1)
        u = 1.0 / (log_scores.exp() @ v.unsqueeze(-1)).squeeze(-1)
    gamma = u.unsqueeze(-1) * log_scores.exp() * v.unsqueeze(-2)

    # Token assignment
    style_tokens: List[str] = []
    content_tokens: List[str] = []
    for i, tok_id in enumerate(tokens[0].tolist()):
        if gamma[0, i, 0] > threshold:
            style_tokens.append(tokenizer.decode(tok_id))
        if gamma[0, i, 1] > threshold:
            content_tokens.append(tokenizer.decode(tok_id))

    style_prompt = " ".join(style_tokens).strip()
    content_prompt = " ".join(content_tokens).strip()
    return PromptOTResult(style_prompt, content_prompt, gamma.detach())


# -----------------------------------------------------------------------------
# Sinkhorn OT-Attention processor
# -----------------------------------------------------------------------------
class SinkhornOTAttnProcessor(nn.Module):
    def __init__(self, head_dim: int, n_iters: int = 5, eps: float = 1e-2):
        super().__init__()
        self.n_iters = n_iters
        self.log_eps = nn.Parameter(torch.log(torch.tensor(eps)))
        self.q_cost = nn.Linear(head_dim, head_dim, bias=False)
        self.k_cost = nn.Linear(head_dim, head_dim, bias=False)

    def _sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_iters):
            scores = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
            scores = scores - torch.logsumexp(scores, dim=-2, keepdim=True)
        return scores.exp()

    def forward(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        residual = hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        qc = self.q_cost(q)
        kc = self.k_cost(k)
        cost = (qc.pow(2).sum(-1, keepdim=True) + kc.pow(2).sum(-1).unsqueeze(-2) - 2 * qc @ kc.transpose(-1, -2))
        log_scores = -cost / torch.exp(self.log_eps)
        if attention_mask is not None:
            log_scores = log_scores + attention_mask
        transport = self._sinkhorn(log_scores)
        hidden_states = transport @ v
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states + residual


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------

def inject_ot_attention(unet: UNet2DConditionModel, n_iters: int = 5, eps: float = 1e-2):
    for name, attn_proc in unet.attn_processors.items():
        if name.endswith("attn2"):
            attn_mod = unet
            for sub in name.split(".")[:-1]:
                attn_mod = getattr(attn_mod, sub)
            head_dim = attn_mod.head_dim if hasattr(attn_mod, "head_dim") else attn_mod.to_q.out_features // attn_mod.num_heads
            attn_mod.set_processor(SinkhornOTAttnProcessor(head_dim, n_iters=n_iters, eps=eps))


def apply_lora(unet: UNet2DConditionModel, rank: int = 8):
    lora_cfg = LoraConfig(r=rank, target_modules=["to_q", "to_v", "q_cost", "k_cost", "log_eps"])
    get_peft_model(unet, lora_cfg)


# -----------------------------------------------------------------------------
# Stage A: Distill OT-Attention to mimic Softmax attention
# -----------------------------------------------------------------------------

def ot_distillation_loop(unet: UNet2DConditionModel, dataloader, optimizer, n_steps: int = 1000):
    unet.train()
    for step, batch in enumerate(dataloader):
        if step >= n_steps:
            break
        noisy_latents, encoder_hidden_states = batch
        preds = unet(noisy_latents, encoder_hidden_states=encoder_hidden_states).sample
        mse = ((unet.attn_processors[list(unet.attn_processors.keys())[0]].last_soft -
                unet.attn_processors[list(unet.attn_processors.keys())[0]].last_ot) ** 2).mean()
        mse.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


# -----------------------------------------------------------------------------
# Stage B: One-shot Style-LoRA fine tuning
# -----------------------------------------------------------------------------

def style_lora_finetune(unet: UNet2DConditionModel, dataloader, optimizer, n_steps: int = 200):
    unet.train()
    for step, batch in enumerate(dataloader):
        if step >= n_steps:
            break
        noisy_latents, noise_pred = batch
        pred = unet(noisy_latents).sample
        loss = ((pred - noise_pred) ** 2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


# -----------------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------------

def load_pipeline(base: str, ot_weights: str | None = None, lora_weights: str | None = None):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
    pipe = StableDiffusionXLPipeline.from_pretrained(base, vae=vae, torch_dtype=torch.float16)
    inject_ot_attention(pipe.unet)
    if ot_weights:
        state = torch.load(ot_weights, map_location="cpu")
        for name, proc in pipe.unet.attn_processors.items():
            if name in state:
                proc.load_state_dict(state[name])
    if lora_weights:
        sd = torch.load(lora_weights, map_location="cpu")
        pipe.unet.load_state_dict(sd, strict=False)
    pipe.to("cuda")
    return pipe


# -----------------------------------------------------------------------------
# Example CLI
# -----------------------------------------------------------------------------

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    text_encoder = StableDiffusionXLPipeline.from_pretrained(args.model, subfolder="text_encoder").text_encoder

    # Prompt OT decomposition
    res = prompt_ot_split(args.prompt, tokenizer, text_encoder)
    print("Style prompt:", res.style_prompt)
    print("Content prompt:", res.content_prompt)

    # Prepare UNet
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")
    inject_ot_attention(unet)
    apply_lora(unet)

    # Dummy dataloaders & optimizers here (placeholders for real code)
    dummy_data = [(
        torch.randn(1, unet.config.in_channels, 64, 64),  # noisy latents
        torch.randn(1, 77, unet.config.cross_attention_dim),  # encoder hidden
    )] * 4
    distill_optim = torch.optim.Adam(unet.parameters(), lr=1e-4)
    ot_distillation_loop(unet, dummy_data, distill_optim, n_steps=4)

    finetune_data = [(
        torch.randn(1, unet.config.in_channels, 64, 64),
        torch.randn(1, unet.config.in_channels, 64, 64),
    )] * 2
    lora_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, unet.parameters()), lr=5e-5)
    style_lora_finetune(unet, finetune_data, lora_optim, n_steps=2)

    # Save weights
    torch.save(unet.state_dict(), args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--tokenizer", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="style_lora.bin")
    main(parser.parse_args())
