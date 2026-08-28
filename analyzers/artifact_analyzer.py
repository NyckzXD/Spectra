import os
import numpy as np
from scipy import ndimage
from PIL import Image


def analyze_artifacts(image_path: str) -> dict:
    """
    Analyzes physical optical artifacts, depth of field gradients, edges, and latent patch boundaries.
    Features evaluated:
    - Optical depth of field & edge sharpness variance (physical lens focus vs synthetic sharpness)
    - Sobel edge orientation coherence & entropy
    - Authentic JPEG 8x8 DCT grid boundary detection (when applicable)
    - Texture variance across spatial blocks
    Returns: {'score': int, 'details': {'metrics': {...}}}
    """
    try:
        img = Image.open(image_path).convert('L')
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape

        if max(h, w) > 1536:
            scale = 1536 / max(h, w)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            h, w = arr.shape

        metrics = {}

        # --- 1. Edge Detection & Sharpness Distribution (Optical Bokeh / DoF) ---
        dx = ndimage.sobel(arr, axis=1)
        dy = ndimage.sobel(arr, axis=0)
        mag = np.hypot(dx, dy)

        edge_threshold = np.percentile(mag, 85)
        strong_edges = mag > edge_threshold
        edge_magnitudes = mag[strong_edges]

        if len(edge_magnitudes) > 100:
            edge_cv = float(np.std(edge_magnitudes) / (np.mean(edge_magnitudes) + 1e-6))
            edge_skew = float(np.mean((edge_magnitudes - np.mean(edge_magnitudes)) ** 3) /
                              (np.std(edge_magnitudes) ** 3 + 1e-6))
        else:
            edge_cv = 0.55
            edge_skew = 0.5

        metrics['edge_sharpness_cv'] = round(edge_cv, 4)
        metrics['edge_sharpness_skew'] = round(edge_skew, 4)

        # --- 2. Edge Orientation Entropy ---
        orientations = np.arctan2(dy, dx)
        strong_orientations = orientations[strong_edges]

        if len(strong_orientations) > 100:
            hist, _ = np.histogram(strong_orientations, bins=36, range=(-np.pi, np.pi))
            hist_norm = hist / np.sum(hist)
            hist_pos = hist_norm[hist_norm > 0]
            edge_orientation_entropy = float(-np.sum(hist_pos * np.log2(hist_pos)))
            orientation_coherence = float(np.var(hist_norm))
        else:
            edge_orientation_entropy = 4.5
            orientation_coherence = 0.001

        metrics['edge_orientation_entropy'] = round(edge_orientation_entropy, 4)
        metrics['orientation_coherence'] = round(orientation_coherence, 6)

        # --- 3. Spatial Texture Variance (Depth & Multi-Layer Scene Complexity) ---
        block_h, block_w = max(h // 4, 1), max(w // 4, 1)
        block_contrasts = []
        for i in range(4):
            for j in range(4):
                y_start = i * block_h
                y_end = min((i + 1) * block_h, h)
                x_start = j * block_w
                x_end = min((j + 1) * block_w, w)
                block = arr[y_start:y_end, x_start:x_end]
                if block.size > 0:
                    block_contrasts.append(float(np.std(block)))

        contrast_var = float(np.var(block_contrasts)) if block_contrasts else 0.0
        contrast_cv = float(np.std(block_contrasts) / (np.mean(block_contrasts) + 1e-6)) if block_contrasts else 0.0
        metrics['texture_variance'] = round(contrast_var, 2)
        metrics['texture_cv'] = round(contrast_cv, 4)

        # --- 4. JPEG 8x8 Grid Boundary Ratio (Only meaningful for original JPEGs) ---
        is_jpeg = False
        ext = os.path.splitext(image_path)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            is_jpeg = True

        jpeg_grid_ratio = 1.0
        if is_jpeg and h >= 32 and w >= 32:
            boundary_diffs = []
            non_boundary_diffs = []

            for col in range(1, w):
                diff = float(np.mean(np.abs(arr[:, col] - arr[:, col - 1])))
                if col % 8 == 0:
                    boundary_diffs.append(diff)
                else:
                    non_boundary_diffs.append(diff)

            for row in range(1, h):
                diff = float(np.mean(np.abs(arr[row, :] - arr[row - 1, :])))
                if row % 8 == 0:
                    boundary_diffs.append(diff)
                else:
                    non_boundary_diffs.append(diff)

            avg_b = np.mean(boundary_diffs) if boundary_diffs else 0
            avg_nb = np.mean(non_boundary_diffs) if non_boundary_diffs else 0

            if avg_nb > 0:
                jpeg_grid_ratio = float(avg_b / avg_nb)

        metrics['jpeg_grid_ratio'] = round(jpeg_grid_ratio, 4)
        metrics['is_jpeg_source'] = is_jpeg

        # --- 5. Symmetrical Calibrated Scoring Model ---
        score = 50.0

        # Optical Depth of Field (Edge CV): Real optical cameras have high edge sharpness variance
        # due to focal plane and bokeh (edge_cv > 0.60). AI images often render uniform sharpness (edge_cv < 0.42).
        if edge_cv > 0.65:
            score -= 16  # Authentic optical lens depth-of-field
        elif edge_cv > 0.52:
            score -= 8
        elif edge_cv < 0.38:
            score += 16  # Unnatural uniform synthetic edge sharpness
        elif edge_cv < 0.46:
            score += 8

        # Spatial Texture Complexity
        if contrast_cv > 0.45:
            score -= 10  # Natural depth and multi-object layering
        elif contrast_cv < 0.18:
            score += 12  # Synthetic flat scene composition

        # Edge Orientation Entropy: Natural photos have rich multi-angle orientations
        if edge_orientation_entropy > 4.8:
            score -= 8   # Rich natural geometric diversity
        elif edge_orientation_entropy < 4.0:
            score += 10  # Biased AI diffusion brush/directionality

        # JPEG Grid: Authentic camera JPEG compression blocks
        if is_jpeg:
            if jpeg_grid_ratio > 1.08:
                score -= 12  # Clear authentic hardware JPEG compression grid
            elif jpeg_grid_ratio < 0.98:
                score += 8   # JPEG file without authentic DCT grid

        final_score = int(round(min(max(score, 0), 100)))

        return {
            'score': final_score,
            'details': {
                'metrics': metrics
            }
        }

    except Exception as e:
        return {
            'score': 50,
            'details': {'metrics': {}, 'error': str(e)}
        }
