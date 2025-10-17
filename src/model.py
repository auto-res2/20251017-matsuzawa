"""Model architectures."""

import torch
from torch import nn
from transformers import DistilBertModel

# ---------------------------------------------------------------------------
# Auxiliary modules
# ---------------------------------------------------------------------------

class HoulsbyAdapter(nn.Module):
    """Implementation of Houlsby adapter (bottleneck)."""

    def __init__(self, hidden_dim: int, reduction_factor: int = 16, non_linearity: str = "relu"):
        super().__init__()
        bottleneck = max(1, hidden_dim // reduction_factor)
        self.down = nn.Linear(hidden_dim, bottleneck)
        self.nonlin = getattr(nn, non_linearity.capitalize())() if hasattr(nn, non_linearity.capitalize()) else nn.ReLU()
        self.up = nn.Linear(bottleneck, hidden_dim)

    def forward(self, x):
        return self.up(self.nonlin(self.down(x)))


class VisionPatchEmbedding(nn.Module):
    """Small ViT-style patch embedding projecting image patches to hidden dim."""

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Linear(in_channels * self.patch_size * self.patch_size, embed_dim)

    def forward(self, imgs):  # imgs: B,C,H,W
        B, C, H, W = imgs.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, "Image dimensions must be divisible by patch size"
        patches = imgs.unfold(2, p, p).unfold(3, p, p)  # B,C,nH,nW,p,p
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # B,N,C,p,p
        patches = patches.view(B, patches.size(1), -1)  # B,N,C*p*p
        return self.proj(patches)  # B, N, D

# ---------------------------------------------------------------------------
# DistilBERT classifier (text or vision)
# ---------------------------------------------------------------------------

class DistilBertClassifier(nn.Module):
    """Adapter-ready DistilBERT classifier supporting vision patch inputs."""

    def __init__(self, model_name: str, num_labels: int, adapter_cfg=None, vision_patch_cfg=None):
        super().__init__()
        self.is_vision = vision_patch_cfg is not None
        self.bert = DistilBertModel.from_pretrained(model_name, cache_dir=".cache/")
        hidden = self.bert.config.hidden_size

        if self.is_vision:
            self.vision = VisionPatchEmbedding(3, vision_patch_cfg.patch_size, hidden)
        else:
            self.vision = None

        if adapter_cfg and adapter_cfg.get("enabled", False):
            self.adapter = HoulsbyAdapter(
                hidden_dim=hidden,
                reduction_factor=int(adapter_cfg.reduction_factor),
                non_linearity=str(adapter_cfg.non_linearity),
            )
        else:
            self.adapter = None

        self.pre_cls = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.cls = nn.Linear(hidden, num_labels)

    def forward(self, batch):
        if self.is_vision and "pixel_values" in batch:
            embeds = self.vision(batch["pixel_values"])  # B,N,D
            mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
            outputs = self.bert(inputs_embeds=embeds, attention_mask=mask)
        else:
            outputs = self.bert(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        pooled = outputs.last_hidden_state[:, 0]  # CLS-token equivalent
        if self.adapter is not None:
            pooled = pooled + self.adapter(pooled)
        x = self.pre_cls(pooled)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.cls(x)
        return logits

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg, tokenizer=None):
    """Factory that builds the appropriate model according to cfg."""
    vision_patch_cfg = (
        cfg.model.vision_patch if str(cfg.model.get("input_representation", "text")) == "image_sequence" else None
    )
    method_tag = str(cfg.method).lower()
    adapter_cfg = cfg.model.task_adapters if method_tag.startswith("proposed") else None
    return DistilBertClassifier(
        model_name=cfg.model.name,
        num_labels=int(cfg.model.num_labels),
        adapter_cfg=adapter_cfg,
        vision_patch_cfg=vision_patch_cfg,
    )
