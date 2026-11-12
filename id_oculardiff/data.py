import os
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import cfg
from .id_tokens import ensure_1d


def repeat_channel(t: torch.Tensor) -> torch.Tensor:
    """Repeat 1-channel tensor to 3 channels."""
    return t.repeat(3, 1, 1)


def load_glasses_set(glasses_file: str) -> set:
    """Read file list with glasses (filenames)."""
    if os.path.exists(glasses_file):
        with open(glasses_file, "r", encoding="utf-8") as f:
            return {line.strip() for line in f.readlines() if line.strip()}
    print(f"[WARN] glasses list not found: {glasses_file}. Glasses filtering disabled.")
    return set()


def extract_subject_id(fname: str) -> int:
    """
    Extract XXX from filename S5XXXRYY.jpg or S5XXXLYY.jpg -> returns int(XXX).
    If format doesn't match, returns -1.
    """
    base = os.path.splitext(fname)[0]
    if len(base) >= 8 and base[0].upper() == "S" and base[1].isdigit():
        sid = base[2:5]
        if sid.isdigit():
            return int(sid)
    return -1


class EyeDataset2Tokens(Dataset):
    """
    Eye dataset that uses ONLY split embeddings:
      base_1.pt and base_2.pt
    and ignores base.pt, even if present.

    train_ocala = False -> slike z očali izloči
    train_ocala = True  -> slike z očali obdrži in jih označi, da lahko kasneje
                           dobijo drug prompt
    """
    def __init__(self, data_root: str, glasses_file: str, resolution: int = 256, train_ocala: bool = False):
        self.image_paths: List[str] = []
        self.pt1_paths: List[str] = []
        self.pt2_paths: List[str] = []
        self.subject_ids: List[int] = []
        self.has_glasses: List[bool] = []

        glasses_set = load_glasses_set(glasses_file)

        for fname in sorted(os.listdir(data_root)):
            if not fname.lower().endswith(".jpg"):
                continue
            if "_seg" in fname:
                continue
            has_glasses = fname in glasses_set
            if (not train_ocala) and has_glasses:
                continue

            base = fname[:-4]
            pt1 = os.path.join(data_root, base + "_1.pt")
            pt2 = os.path.join(data_root, base + "_2.pt")


            if not (os.path.exists(pt1) and os.path.exists(pt2)):
                continue

            self.image_paths.append(os.path.join(data_root, fname))
            self.pt1_paths.append(pt1)
            self.pt2_paths.append(pt2)
            self.subject_ids.append(extract_subject_id(fname))
            self.has_glasses.append(has_glasses)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                "EyeDataset2Tokens: no training samples found.\n"
                "Make sure its ran the split script so every image has *_1.pt and *_2.pt,\n"
                "and that glasses filtering / inclusion settings did not remove everything."
            )


        longer = resolution
        shorter = int(resolution * 3 / 4)
        self.resize_h, self.resize_w = (shorter, longer) if longer > shorter else (longer, shorter)

        self.tform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.resize_h, self.resize_w)),
            transforms.ToTensor(),
            transforms.Lambda(repeat_channel),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        id1 = ensure_1d(torch.load(self.pt1_paths[idx], map_location="cpu"))
        id2 = ensure_1d(torch.load(self.pt2_paths[idx], map_location="cpu"))

        if id1.numel() != cfg.id_token_dim or id2.numel() != cfg.id_token_dim:
            raise ValueError(
                f"Bad embedding dim for {os.path.basename(self.image_paths[idx])}: "
                f"id1={id1.numel()}, id2={id2.numel()}, expected={cfg.id_token_dim}"
            )

        id_tokens = torch.stack([id1, id2], dim=0)
        subj = torch.tensor(self.subject_ids[idx], dtype=torch.int64)

        has_glasses = torch.tensor(self.has_glasses[idx], dtype=torch.bool)

        return {
            "images": self.tform(img),
            "id_tokens": id_tokens,
            "subject": subj,
            "has_glasses": has_glasses,
        }


def collate_fn(batch):
    imgs = torch.stack([b["images"] for b in batch])
    ids = torch.stack([b["id_tokens"] for b in batch])
    subj = torch.stack([b["subject"] for b in batch])
    has_glasses = torch.stack([b["has_glasses"] for b in batch])
    return {"images": imgs, "id_tokens": ids, "subject": subj, "has_glasses": has_glasses}
