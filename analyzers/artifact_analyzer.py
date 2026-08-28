import numpy as np
from scipy import ndimage
from PIL import Image


def analyze_artifacts(image_path: str) -> dict:
    """
    Analyzes visual artifacts, edges, textures, and structural patterns.
    Detects JPEG grid misalignment, checkerboard artifacts (GAN), and edge sharpness distribution.
    Returns: {'score': int, 'details': {'metrics': {...}}}
    """
    try:
        img = Image.open(image_path).convert('L')
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape

        metrics = {}

        # --- Sobel Edge Detection ---
        dx = ndimage.sobel(arr, axis=1)
        dy = ndimage.sobel(arr, axis=0)
        mag = np.hypot(dx, dy)

        edge_threshold = np.percentile(mag, 85)
        strong_edges = mag > edge_threshold
        edge_density = float(np.mean(strong_edges))
        metrics['edge_density'] = round(edge_density, 6)

        # --- Edge orientation coherence ---
        orientations = np.arctan2(dy, dx)
        strong_orientations = orientations[strong_edges]

        if len(strong_orientations) > 0:
            hist, _ = np.histogram(strong_orientations, bins=36, range=(-np.pi, np.pi))
            hist_norm = hist / np.sum(hist)
            coherence = float(np.var(hist_norm))

            # Edge orientation entropy
            hist_pos = hist_norm[hist_norm > 0]
            edge_orientation_entropy = float(-np.sum(hist_pos * np.log2(hist_pos)))
        else:
            coherence = 0.0
            edge_orientation_entropy = 0.0

        metrics['edge_coherence'] = round(coherence, 6)
        metrics['edge_orientation_entropy'] = round(edge_orientation_entropy, 4)

        # --- Edge sharpness distribution ---
        # GANs often produce edges with unnaturally uniform sharpness
        edge_magnitudes = mag[strong_edges]
        if len(edge_magnitudes) > 100:
            edge_cv = float(np.std(edge_magnitudes) / (np.mean(edge_magnitudes) + 1e-6))
            edge_skew = float(np.mean((edge_magnitudes - np.mean(edge_magnitudes)) ** 3) /
                              (np.std(edge_magnitudes) ** 3 + 1e-6))
        else:
            edge_cv = 0.5
            edge_skew = 0.0

        metrics['edge_sharpness_cv'] = round(edge_cv, 4)
        metrics['edge_sharpness_skew'] = round(edge_skew, 4)

        # --- Block texture variance (4x4 grid) ---
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
        metrics['texture_variance'] = round(contrast_var, 2)

        # --- Laplacian variance (focus/blur indicator) ---
        lap = ndimage.laplace(arr)
        lap_var = float(np.var(lap))
        metrics['laplacian_variance'] = round(lap_var, 2)

        # --- JPEG grid alignment detection ---
        # Real JPEG images have compression artifacts aligned to 8x8 blocks
        # AI-generated images (even saved as JPEG) may not have this alignment
        jpeg_grid_score = 0.0
        if h >= 16 and w >= 16:
            # Compute average absolute difference at 8-pixel boundaries vs. non-boundaries
            # Horizontal boundaries
            boundary_diffs_h = []
            non_boundary_diffs_h = []
            for col in range(1, w):
                diff = float(np.mean(np.abs(arr[:, col].astype(np.float64) -
                                            arr[:, col - 1].astype(np.float64))))
                if col % 8 == 0:
                    boundary_diffs_h.append(diff)
                else:
                    non_boundary_diffs_h.append(diff)

            # Vertical boundaries
            boundary_diffs_v = []
            non_boundary_diffs_v = []
            for row in range(1, h):
                diff = float(np.mean(np.abs(arr[row, :].astype(np.float64) -
                                            arr[row - 1, :].astype(np.float64))))
                if row % 8 == 0:
                    boundary_diffs_v.append(diff)
                else:
                    non_boundary_diffs_v.append(diff)

            avg_boundary = np.mean(boundary_diffs_h + boundary_diffs_v) if (boundary_diffs_h + boundary_diffs_v) else 0
            avg_non_boundary = np.mean(non_boundary_diffs_h + non_boundary_diffs_v) if (non_boundary_diffs_h + non_boundary_diffs_v) else 0

            if avg_non_boundary > 0:
                jpeg_grid_ratio = avg_boundary / avg_non_boundary
                # Ratio > 1 suggests JPEG grid artifacts (real JPEG has stronger boundaries)
                jpeg_grid_score = float(jpeg_grid_ratio)
            else:
                jpeg_grid_score = 1.0

        metrics['jpeg_grid_ratio'] = round(jpeg_grid_score, 4)

        # --- Checkerboard artifact detection (GAN upsampling fingerprint) ---
        # GANs using transposed convolutions produce checkerboard patterns
        # Detect by analyzing 2x2 pixel patterns
        checkerboard_score = 0.0
        if h >= 8 and w >= 8:
            # Compute differences in a 2x2 checkerboard pattern
            even_even = arr[0::2, 0::2]
            even_odd = arr[0::2, 1::2]
            odd_even = arr[1::2, 0::2]
            odd_odd = arr[1::2, 1::2]

            min_h = min(even_even.shape[0], even_odd.shape[0], odd_even.shape[0], odd_odd.shape[0])
            min_w = min(even_even.shape[1], even_odd.shape[1], odd_even.shape[1], odd_odd.shape[1])

            ee = even_even[:min_h, :min_w].astype(np.float64)
            eo = even_odd[:min_h, :min_w].astype(np.float64)
            oe = odd_even[:min_h, :min_w].astype(np.float64)
            oo = odd_odd[:min_h, :min_w].astype(np.float64)

            # Checkerboard = diagonal pairs more similar than adjacent pairs
            diag_diff = np.mean(np.abs(ee - oo) + np.abs(eo - oe))
            adj_diff = np.mean(np.abs(ee - eo) + np.abs(ee - oe))

            if adj_diff > 0:
                checkerboard_ratio = diag_diff / adj_diff
                # If ratio is very different from 1.0, checkerboard pattern present
                checkerboard_score = float(abs(1.0 - checkerboard_ratio))

        metrics['checkerboard_score'] = round(checkerboard_score, 4)

        # --- Scoring ---
        score = 50

        # Low texture variance → AI (uniform texture)
        if contrast_var < 100:
            score += 15
        elif contrast_var > 500:
            score -= 12

        # High edge coherence (biased orientations) → AI
        if coherence > 0.005:
            score += 12
        elif coherence < 0.001:
            score -= 5

        # Low edge sharpness CV → AI (unnaturally uniform sharpness)
        if edge_cv < 0.4:
            score += 10
        elif edge_cv > 0.8:
            score -= 8

        # Low laplacian variance → blurry/smooth → AI tendency
        if lap_var < 50:
            score += 12
        elif lap_var > 200:
            score -= 8

        # JPEG grid: ratio near 1.0 = no JPEG grid = likely not from camera JPEG
        if 0.95 <= jpeg_grid_score <= 1.05:
            score += 8  # No JPEG artifacts → more likely AI
        elif jpeg_grid_score > 1.15:
            score -= 10  # Strong JPEG grid → real JPEG from camera

        # Checkerboard artifacts
        if checkerboard_score > 0.1:
            score += min(int(checkerboard_score * 30), 20)

        return {
            'score': min(max(int(score), 0), 100),
            'details': {
                'metrics': metrics
            }
        }

    except Exception as e:
        return {
            'score': 50,
            'details': {'metrics': {}}
        }
