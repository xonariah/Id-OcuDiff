from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .generation_helpers import build_scheduler


def load_trusted_local_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class PixelClassifierMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512), nn.Dropout(p=0.10),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.BatchNorm1d(256), nn.Dropout(p=0.10),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.BatchNorm1d(128),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.layers(x)


def load_ensemble(manifest_path, device):
    manifest_path = Path(manifest_path)
    manifest = load_trusted_local_checkpoint(manifest_path)
    input_dim = int(manifest["input_dim"])
    num_classes = int(manifest["num_classes"])
    if num_classes != 4:
        raise ValueError(f"Pričakujem 4 razrede, manifest vsebuje {num_classes}.")
    mean = manifest["x_mean"].to(device).float()
    std = manifest["x_std"].to(device).float()
    channel_indices = manifest.get("feature_channel_indices")
    if channel_indices is not None:
        channel_indices = torch.as_tensor(channel_indices, dtype=torch.long, device="cpu")
    models = []
    model_paths = []
    for original_path in manifest["model_paths"]:
        model_name = str(original_path).replace("\\", "/").rsplit("/", 1)[-1]
        model_path = manifest_path.parent / "models" / model_name
        if not model_path.is_file():
            raise FileNotFoundError(f"Ne najdem člana ansambla: {model_path}")
        model_ckpt = load_trusted_local_checkpoint(model_path)
        model = PixelClassifierMLP(input_dim, num_classes).to(device)
        model.load_state_dict(model_ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        model_paths.append(str(model_path))
    if len(models) != 10:
        raise RuntimeError(f"Pričakujem 10 naučenih modelov, naloženih: {len(models)}.")
    return {
        "models": models,
        "model_paths": model_paths,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "mean": mean,
        "std": std,
        "channel_indices": channel_indices,
        "manifest": manifest,
    }


def map_feature_timesteps(ensemble, components, device, settings):
    manifest = ensemble["manifest"]
    train_indices = [int(x) for x in manifest["feature_timestep_indices"]]
    train_num_steps = int(manifest.get("num_inference_steps", 25))
    ref_scheduler = build_scheduler(settings)
    ref_scheduler.set_timesteps(train_num_steps, device=device)
    ref_sigmas = ref_scheduler.sigmas.detach().float().cpu().numpy()
    new_scheduler = components["scheduler"]
    new_scheduler.set_timesteps(settings.sample_inference_steps, device=device)
    new_sigmas = new_scheduler.sigmas.detach().float().cpu().numpy()
    indices = [
        int(np.argmin(np.abs(new_sigmas[:settings.sample_inference_steps] - ref_sigmas[i])))
        for i in train_indices
    ]
    if len(set(indices)) != len(indices):
        raise RuntimeError("Preslikava časovnih korakov je dala podvojene indekse.")
    return list(manifest["feature_hook_blocks"]), indices


def ensemble_compare_predict_mean_prob_maps_from_feature_map(
    feature_map: torch.Tensor,
    ensemble,
    device,
    batch_size: int = 8192,
):
    ensemble_loaded_input_dim = ensemble["input_dim"]
    ensemble_loaded_num_classes = ensemble["num_classes"]
    ensemble_x_mean_t = ensemble["mean"]
    ensemble_x_std_t = ensemble["std"]
    ensemble_mlps = ensemble["models"]
    c, h, w = feature_map.shape

    if c != ensemble_loaded_input_dim:
        raise RuntimeError(
            f"Feature dim mismatch: feature_map ima {c}, ensemble MLP pričakuje {ensemble_loaded_input_dim}."
        )

    x = (
        feature_map
        .permute(1, 2, 0)
        .reshape(-1, c)
        .contiguous()
    )

    prob_chunks = []

    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size].to(device).float()
        xb = (xb - ensemble_x_mean_t) / ensemble_x_std_t

        prob_sum = None

        for model in ensemble_mlps:
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)

            if prob_sum is None:
                prob_sum = probs
            else:
                prob_sum = prob_sum + probs

        mean_probs = prob_sum / float(len(ensemble_mlps))
        prob_chunks.append(mean_probs.detach().cpu().to(torch.float16))

    probs_flat = torch.cat(prob_chunks, dim=0).float().numpy()
    prob_maps = probs_flat.reshape(h, w, ensemble_loaded_num_classes).transpose(2, 0, 1)

    return prob_maps.astype(np.float32)
