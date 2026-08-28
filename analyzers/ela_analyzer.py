import io
import base64
import numpy as np
from PIL import Image


def _viridis_colormap(value):
    """Vectorized viridis-like LUT for ELA heatmaps."""
    r = np.clip(np.where(value < 0.5, value * 0.2, 0.1 + (value - 0.5) * 1.8), 0, 1)
    g = np.clip(np.where(value < 0.25, value * 0.4,
                np.where(value < 0.75, 0.1 + (value - 0.25) * 1.6, 0.9 - (value - 0.75) * 0.4)), 0, 1)
    b = np.clip(np.where(value < 0.5, 0.3 + value * 1.0, 0.8 - (value - 0.5) * 1.6), 0, 1)
    return r, g, b


def colorize_heatmap(gray_arr):
    """Applies a viridis-like colormap to a grayscale numpy array."""
    normalized = gray_arr.astype(np.float32) / 255.0
    r, g, b = _viridis_colormap(normalized)
    colored = np.stack([
        (r * 255).astype(np.uint8),
        (g * 255).astype(np.uint8),
        (b * 255).astype(np.uint8)
    ], axis=-1)
    return Image.fromarray(colored)


def analyze_ela(image_path: str) -> dict:
    """
    Performs Error Level Analysis (ELA) across multi-scale compression levels.
    Evaluates:
    - Consistency of compression error with underlying scene textures
    - Spatial uniformity and quadrant coefficient of variation
    - Multi-generational compression anomalies
    Returns: {'score': int, 'details': {'metrics': {...}}, 'visualization': 'base64string'}
    """
    try:
        original = Image.open(image_path).convert('RGB')
        orig_arr = np.array(original, dtype=np.float32)
        h, w, _ = orig_arr.shape

        if max(h, w) > 1536:
            scale = 1536 / max(h, w)
            original = original.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            orig_arr = np.array(original, dtype=np.float32)
            h, w, _ = orig_arr.shape

        qualities = [95, 90, 85, 75]
        diffs = []

        for q in qualities:
            temp_io = io.BytesIO()
            original.save(temp_io, 'JPEG', quality=q)
            temp_io.seek(0)
            compressed = Image.open(temp_io).convert('RGB')

            diff = np.abs(orig_arr - np.array(compressed, dtype=np.float32))
            gray_diff = np.mean(diff, axis=2)
            diffs.append(gray_diff)

        avg_diff = np.mean(diffs, axis=0)

        # Normalize difference for metrics
        mean_ela = float(np.mean(avg_diff))
        std_ela = float(np.std(avg_diff))
        max_ela = float(np.max(avg_diff))
        uniformity_cv = float(std_ela / (mean_ela + 1e-6))

        # --- 1. Texture-to-Error Concordance ---
        # Real optical photos: areas with high visual texture (edges/fine details) have significantly
        # higher ELA error than smooth flat sky/walls.
        # AI images (especially diffusion) often have abnormally uniform error distribution.
        gray_orig = 0.299 * orig_arr[:, :, 0] + 0.587 * orig_arr[:, :, 1] + 0.114 * orig_arr[:, :, 2]
        grad_y, grad_x = np.gradient(gray_orig)
        texture_energy = np.hypot(grad_x, grad_y)

        # High texture vs flat regions
        high_tex_mask = texture_energy > np.percentile(texture_energy, 75)
        flat_tex_mask = texture_energy < np.percentile(texture_energy, 25)

        ela_high_tex = np.mean(avg_diff[high_tex_mask]) if np.any(high_tex_mask) else 1.0
        ela_flat_tex = np.mean(avg_diff[flat_tex_mask]) if np.any(flat_tex_mask) else 1.0
        texture_error_ratio = float(ela_high_tex / (ela_flat_tex + 1e-6))

        # --- 2. Quadrant Variance & Block Uniformity ---
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            avg_diff[:mid_h, :mid_w],
            avg_diff[:mid_h, mid_w:],
            avg_diff[mid_h:, :mid_w],
            avg_diff[mid_h:, mid_w:]
        ]
        quadrant_means = [float(np.mean(q)) for q in quadrants]
        quadrant_cv = float(np.std(quadrant_means) / (np.mean(quadrant_means) + 1e-6))

        # 8x8 block variance
        block_size = 8
        block_means = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = avg_diff[y:y + block_size, x:x + block_size]
                block_means.append(np.mean(block))

        block_variance = float(np.var(block_means)) if block_means else 0.0

        # --- 3. Symmetrical Calibrated Scoring Model ---
        score = 50.0

        # Texture-to-Error ratio: Real photos have distinct texture-dependent compression response (> 2.0)
        if texture_error_ratio > 2.8:
            score -= 16  # High natural texture-error concordance
        elif texture_error_ratio > 2.0:
            score -= 8
        elif texture_error_ratio < 1.3:
            score += 16  # Synthetic uniform error independence
        elif texture_error_ratio < 1.6:
            score += 8

        # Quadrant variation: Natural scenes have lighting/focus gradients across quadrants
        if quadrant_cv > 0.35:
            score -= 10  # Natural spatial variation
        elif quadrant_cv < 0.08:
            score += 12  # Synthetic homogeneous spatial composition

        # Uniformity: Overly uniform low-noise ELA indicates direct synthetic synthesis
        if uniformity_cv < 0.45 and mean_ela < 3.0:
            score += 14  # Direct digital synthesis
        elif uniformity_cv > 0.85:
            score -= 8   # Realistic mixed-content photographic response

        final_score = int(round(min(max(score, 0), 100)))

        # --- 4. Visualization ---
        max_val = avg_diff.max()
        vis_diff = (avg_diff / (max_val + 1e-6) * 255.0) if max_val > 0 else avg_diff
        ela_img = colorize_heatmap(vis_diff.astype(np.uint8))

        buffered = io.BytesIO()
        ela_img.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            'score': final_score,
            'details': {
                'metrics': {
                    'mean_ela': round(mean_ela, 2),
                    'std_ela': round(std_ela, 2),
                    'texture_error_ratio': round(texture_error_ratio, 4),
                    'quadrant_cv': round(quadrant_cv, 4),
                    'uniformity_cv': round(uniformity_cv, 4),
                    'block_variance': round(block_variance, 2),
                }
            },
            'visualization': img_str
        }

    except Exception as e:
        return {
            'score': 50,
            'details': {'metrics': {}, 'error': str(e)},
            'visualization': ''
        }
