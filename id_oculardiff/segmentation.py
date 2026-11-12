import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .ensemble import ensemble_compare_predict_mean_prob_maps_from_feature_map
from .morphology import ensemble_compare_regularize_label_mask


def ensemble_compare_smooth_prob_maps(prob_maps: np.ndarray, sigma: float = 0.8):
    prob_maps = prob_maps.astype(np.float32, copy=True)

    if sigma is None or sigma <= 0:
        return prob_maps

    for c in range(prob_maps.shape[0]):
        prob_maps[c] = cv2.GaussianBlur(
            prob_maps[c],
            ksize=(0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REPLICATE,
        )

    denom = prob_maps.sum(axis=0, keepdims=True)
    return prob_maps / np.maximum(denom, 1e-8)


def ensemble_compare_soft_upsample_prob_maps_to_label(prob_maps: np.ndarray, out_width: int, out_height: int):
    probs_t = torch.from_numpy(prob_maps).unsqueeze(0).float()

    probs_up = F.interpolate(
        probs_t,
        size=(out_height, out_width),
        mode="bilinear",
        align_corners=False,
    )[0]

    return probs_up.argmax(dim=0).cpu().numpy().astype(np.uint8)


def segment_feature_map(feature_map, ensemble, device, width, height):
    probability_maps = ensemble_compare_predict_mean_prob_maps_from_feature_map(
        feature_map, ensemble, device, batch_size=8192,
    )
    probability_maps = ensemble_compare_smooth_prob_maps(probability_maps, sigma=0.8)
    labels = ensemble_compare_soft_upsample_prob_maps_to_label(
        probability_maps, out_width=width, out_height=height,
    )
    return ensemble_compare_regularize_label_mask(labels)
