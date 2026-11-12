import cv2
import numpy as np

EYE_BOUNDARY_SIGMA = 4.0
USE_ELLIPSE_IRIS = False
USE_ELLIPSE_PUPIL = True
IRIS_ELLIPSE_SCALE = 1.00
PUPIL_ELLIPSE_SCALE = 0.98
EYE_CLOSE_KSIZE, EYE_OPEN_KSIZE = 9, 3
IRIS_CLOSE_KSIZE, IRIS_OPEN_KSIZE = 5, 3
PUPIL_CLOSE_KSIZE, PUPIL_OPEN_KSIZE = 5, 3
MIN_EYE_AREA, MIN_IRIS_AREA, MIN_PUPIL_AREA = 1000, 200, 50


def ensemble_compare_largest_component(mask_bool: np.ndarray, min_area: int = 0):
    mask_u8 = mask_bool.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

    if num_labels <= 1:
        return np.zeros_like(mask_bool, dtype=bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    best_area = int(stats[best_idx, cv2.CC_STAT_AREA])

    if best_area < min_area:
        return np.zeros_like(mask_bool, dtype=bool)

    return labels == best_idx


def ensemble_compare_fill_holes(mask_bool: np.ndarray):
    mask_u8 = mask_bool.astype(np.uint8) * 255

    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, holes)

    return filled > 0


def ensemble_compare_morph_clean(
    mask_bool: np.ndarray,
    close_ksize: int = 0,
    open_ksize: int = 0,
    keep_largest: bool = True,
    fill: bool = True,
    min_area: int = 0,
):
    m = mask_bool.astype(np.uint8) * 255

    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(close_ksize), int(close_ksize)),
        )
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    if open_ksize and open_ksize > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(open_ksize), int(open_ksize)),
        )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    out = m > 0

    if keep_largest:
        out = ensemble_compare_largest_component(out, min_area=min_area)

    if fill:
        out = ensemble_compare_fill_holes(out)

    return out


def ensemble_compare_sdf_smooth_binary(
    mask_bool: np.ndarray,
    sigma: float = 4.0,
    preserve_area: bool = True,
    keep_largest: bool = True,
    fill: bool = True,
    min_area: int = 0,
):
    if mask_bool.sum() == 0:
        return mask_bool.astype(bool)

    m = mask_bool.astype(np.uint8)

    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    sdf = dist_in - dist_out

    if sigma is not None and sigma > 0:
        sdf = cv2.GaussianBlur(
            sdf,
            ksize=(0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REPLICATE,
        )

    if preserve_area:
        target_area = int(mask_bool.sum())
        flat = sdf.reshape(-1)

        if 0 < target_area < flat.size:
            threshold = np.partition(flat, flat.size - target_area)[flat.size - target_area]
            out = sdf >= threshold
        else:
            out = sdf > 0
    else:
        out = sdf > 0

    if keep_largest:
        out = ensemble_compare_largest_component(out, min_area=min_area)

    if fill:
        out = ensemble_compare_fill_holes(out)

    return out


def ensemble_compare_fit_ellipse_mask(
    mask_bool: np.ndarray,
    output_shape,
    scale: float = 1.0,
    min_area: int = 0,
):
    h, w = output_shape

    clean = ensemble_compare_largest_component(mask_bool, min_area=min_area)
    clean = ensemble_compare_fill_holes(clean)

    if clean.sum() < min_area:
        return clean

    contours, _ = cv2.findContours(
        clean.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        return clean

    cnt = max(contours, key=cv2.contourArea)

    if len(cnt) < 5:
        return clean

    try:
        ellipse = cv2.fitEllipse(cnt)
    except cv2.error:
        return clean

    (cx, cy), (axis_a, axis_b), angle = ellipse

    axis_a = max(1.0, axis_a * float(scale))
    axis_b = max(1.0, axis_b * float(scale))

    out = np.zeros((h, w), dtype=np.uint8)

    cv2.ellipse(
        out,
        ((float(cx), float(cy)), (float(axis_a), float(axis_b)), float(angle)),
        color=255,
        thickness=-1,
    )

    return out > 0


def ensemble_compare_regularize_label_mask(label_mask: np.ndarray):
    label_mask = label_mask.astype(np.uint8)
    h, w = label_mask.shape

    eye_raw = label_mask > 0

    eye = ensemble_compare_morph_clean(
        eye_raw,
        close_ksize=EYE_CLOSE_KSIZE,
        open_ksize=EYE_OPEN_KSIZE,
        keep_largest=True,
        fill=True,
        min_area=MIN_EYE_AREA,
    )

    eye = ensemble_compare_sdf_smooth_binary(
        eye,
        sigma=EYE_BOUNDARY_SIGMA,
        preserve_area=True,
        keep_largest=True,
        fill=True,
        min_area=MIN_EYE_AREA,
    )

    iris_raw = np.isin(label_mask, [2, 3])

    iris_clean = ensemble_compare_morph_clean(
        iris_raw,
        close_ksize=IRIS_CLOSE_KSIZE,
        open_ksize=IRIS_OPEN_KSIZE,
        keep_largest=True,
        fill=True,
        min_area=MIN_IRIS_AREA,
    )

    if USE_ELLIPSE_IRIS:
        iris = ensemble_compare_fit_ellipse_mask(
            iris_clean,
            output_shape=(h, w),
            scale=IRIS_ELLIPSE_SCALE,
            min_area=MIN_IRIS_AREA,
        )
    else:
        iris = ensemble_compare_sdf_smooth_binary(
            iris_clean,
            sigma=3.0,
            preserve_area=True,
            keep_largest=True,
            fill=True,
            min_area=MIN_IRIS_AREA,
        )

    iris = iris & eye

    pupil_raw = label_mask == 3

    pupil_clean = ensemble_compare_morph_clean(
        pupil_raw,
        close_ksize=PUPIL_CLOSE_KSIZE,
        open_ksize=PUPIL_OPEN_KSIZE,
        keep_largest=True,
        fill=True,
        min_area=MIN_PUPIL_AREA,
    )

    if USE_ELLIPSE_PUPIL:
        pupil = ensemble_compare_fit_ellipse_mask(
            pupil_clean,
            output_shape=(h, w),
            scale=PUPIL_ELLIPSE_SCALE,
            min_area=MIN_PUPIL_AREA,
        )
    else:
        pupil = ensemble_compare_sdf_smooth_binary(
            pupil_clean,
            sigma=2.0,
            preserve_area=True,
            keep_largest=True,
            fill=True,
            min_area=MIN_PUPIL_AREA,
        )

    pupil = pupil & iris
    iris = iris | pupil

    out = np.zeros_like(label_mask, dtype=np.uint8)
    out[eye] = 1
    out[iris] = 2
    out[pupil] = 3

    return out
