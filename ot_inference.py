import argparse
import os

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel


class SinkhornOTAttnProcessor(nn.Module):
    """Sinkhorn Optimal Transport attention processor."""

    def __init__(self, head_dim: int, n_iters: int = 5, eps: float = 1e-2):
        super().__init__()
        self.n_iters = n_iters
        self.log_eps = nn.Parameter(torch.log(torch.tensor(eps)))
        self.q_cost = nn.Linear(head_dim, head_dim, bias=False)
        self.k_cost = nn.Linear(head_dim, head_dim, bias=False)
        self.last_ot: torch.Tensor | None = None
        self.last_soft: torch.Tensor | None = None

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
        cost = (
            qc.pow(2).sum(-1, keepdim=True)
            + kc.pow(2).sum(-1).unsqueeze(-2)
            - 2 * qc @ kc.transpose(-1, -2)
        )
        log_scores = -cost / torch.exp(self.log_eps)
        if attention_mask is not None:
            log_scores = log_scores + attention_mask

        self.last_soft = torch.softmax(log_scores, dim=-1)
        transport = self._sinkhorn(log_scores)
        self.last_ot = transport

        hidden_states = transport @ v
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states + residual


def inject_ot_attention(unet: UNet2DConditionModel, n_iters: int = 5, eps: float = 1e-2):
    for name, _ in unet.attn_processors.items():
        if name.endswith("attn2"):
            attn_module = unet
            for sub in name.split(".")[:-1]:
                attn_module = getattr(attn_module, sub)
            head_dim = getattr(attn_module, "head_dim", None)
            if head_dim is None:
                head_dim = attn_module.to_q.out_features // attn_module.num_heads
            attn_module.set_processor(SinkhornOTAttnProcessor(head_dim, n_iters=n_iters, eps=eps))


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


def parse_args():
    parser = argparse.ArgumentParser(description="OT-attention inference script")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--ot_weights", type=str, default=None)
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="ot_outputs")
    return parser.parse_args()


def main(args):
    pipe = load_pipeline(args.model, args.ot_weights, args.lora_weights)
    os.makedirs(args.output_dir, exist_ok=True)
    images = pipe(args.prompt, num_images_per_prompt=args.num_images).images
    for i, img in enumerate(images):
        img.save(os.path.join(args.output_dir, f"img_{i}.png"))


if __name__ == "__main__":
    main(parse_args())
