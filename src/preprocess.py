"""Dataset preprocessing and DataLoader construction."""
from typing import Tuple, List
import random

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets as tv_datasets, transforms
from transformers import AutoTokenizer
from datasets import load_dataset

# ----------------------------------------------------------------------------
# CIFAR-10 helpers
# ----------------------------------------------------------------------------

class _CIFARWrapper(torch.utils.data.Dataset):
    """Wrap torchvision dataset to return dicts compatible with model forward."""

    def __init__(self, base, tfm):
        self.base = base
        self.tfm = tfm

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = self.tfm(img)
        return {"pixel_values": img, "labels": torch.tensor(label, dtype=torch.long)}


def _split(dataset: torch.utils.data.Dataset, splits: Tuple[float, float, float]):
    n_total = len(dataset)
    lengths = [int(p * n_total) for p in splits]
    lengths[-1] = n_total - sum(lengths[:-1])  # ensure all data used
    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))


def _build_cifar10(cfg, trial_mode: bool):
    size = int(cfg.dataset.preprocessing.resize)
    aug: List[str] = list(cfg.dataset.preprocessing.augmentations)

    tfm_train = [
        transforms.RandomCrop(size, padding=4) if "random_crop" in aug else transforms.Resize(size),
        transforms.RandomHorizontalFlip() if "horizontal_flip" in aug else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ]
    tfm_eval = [transforms.Resize(size), transforms.ToTensor()]

    hf_dataset = load_dataset("uoft-cs/cifar10", split="train", cache_dir=".cache/")
    
    class _HFCIFARWrapper(torch.utils.data.Dataset):
        def __init__(self, indices, full_data, tfm):
            self.indices = indices
            self.data = full_data
            self.tfm = tfm
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            actual_idx = self.indices[idx]
            item = self.data[actual_idx]
            img = item['img']
            label = item['label']
            img = self.tfm(img)
            return {"pixel_values": img, "labels": torch.tensor(label, dtype=torch.long)}
    
    n_total = len(hf_dataset)
    splits = (cfg.dataset.split.train, cfg.dataset.split.val, cfg.dataset.split.test)
    lengths = [int(p * n_total) for p in splits]
    lengths[-1] = n_total - sum(lengths[:-1])
    
    indices = list(range(n_total))
    random.Random(42).shuffle(indices)
    train_indices = indices[:lengths[0]]
    val_indices = indices[lengths[0]:lengths[0]+lengths[1]]
    test_indices = indices[lengths[0]+lengths[1]:]

    train_set = _HFCIFARWrapper(train_indices, hf_dataset, transforms.Compose(tfm_train))
    val_set = _HFCIFARWrapper(val_indices, hf_dataset, transforms.Compose(tfm_eval))
    test_set = _HFCIFARWrapper(test_indices, hf_dataset, transforms.Compose(tfm_eval))

    bs = 2 if trial_mode else int(cfg.training.batch_size)
    dl_kwargs = dict(batch_size=bs, num_workers=2, pin_memory=True)
    return (
        DataLoader(train_set, shuffle=True, **dl_kwargs),
        DataLoader(val_set, shuffle=False, **dl_kwargs),
        DataLoader(test_set, shuffle=False, **dl_kwargs),
        None,
    )

# ----------------------------------------------------------------------------
# Alpaca-cleaned helpers (binary classification: translation vs others)
# ----------------------------------------------------------------------------

def _label_from_instruction(instr: str) -> int:
    lowered = instr.lower()
    keywords = ["translate", "translation", "translating", "translator"]
    return int(any(k in lowered for k in keywords))


def _encode_text(batch, tokenizer, max_len):
    instructions = batch["instruction"]
    inputs = batch.get("input", ["" for _ in range(len(instructions))])
    texts = [f"Instruction: {ins} Input: {inp}" for ins, inp in zip(instructions, inputs)]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)
    enc["labels"] = [_label_from_instruction(ins) for ins in instructions]
    return enc


def _build_alpaca(cfg, trial_mode: bool):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=".cache/")
    raw = load_dataset("yahma/alpaca-cleaned", split="train", cache_dir=".cache/")

    encoded = raw.map(
        lambda b: _encode_text(b, tokenizer, cfg.dataset.max_length),
        batched=True,
        remove_columns=raw.column_names,
    )
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    total = len(encoded)
    train_size = int(cfg.dataset.split.train * total)
    val_size = int(cfg.dataset.split.val * total)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        encoded,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    bs = 2 if trial_mode else int(cfg.training.batch_size)
    dl_kwargs = dict(batch_size=bs, num_workers=2, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **dl_kwargs),
        DataLoader(val_ds, shuffle=False, **dl_kwargs),
        DataLoader(test_ds, shuffle=False, **dl_kwargs),
        tokenizer,
    )

# ----------------------------------------------------------------------------
# Public factory
# ----------------------------------------------------------------------------

def build_dataloaders(cfg, trial_mode: bool = False):
    name = str(cfg.dataset.name).lower()
    if name == "cifar10":
        return _build_cifar10(cfg, trial_mode)
    elif name == "alpaca-cleaned":
        return _build_alpaca(cfg, trial_mode)
    else:
        raise ValueError(f"Unsupported dataset '{cfg.dataset.name}'")
