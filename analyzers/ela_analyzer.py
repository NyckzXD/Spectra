import io
import base64
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def _viridis_colormap(value):
    """Attempt a simple viridis-like LUT for ELA heatmaps (vectorized)."""
    # Attempt an approximation: dark purple → blue → green → yellow
    r = np.clip(np.where(value < 0.5, value * 0.2, 0.1 + (value - 0.5) * 1.8), 0, 1)
    g = np.clip(np.where(value < 0.25, value * 0.4,
                np.where(value < 0.75, 0.1 + (value - 0.25) * 1.6, 0.9 - (value - 0.75) * 0.4)), 0, 1)
    b = np.clip(np.where(value < 0.5, 0.3 + value * 1.0, 0.8 - (value - 0.5) * 1.6), 0, 1)
    return r, g, b


def colorize_heatmap(gray_arr):
    """Applies a viridis-like colormap to a grayscale numpy array. Fully vectorized."""
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
    Performs Error Level Analysis at multiple JPEG qualities.
    Detects compression inconsistencies, ghost JPEG artifacts, and uniformity anomalies.
    Returns: {'score': int, 'details': {'metrics': {...}}, 'visualization': 'base64string'}
    """
    try:
        original = Image.open(image_path).convert('RGB')
        orig_arr = np.array(original, dtype=np.float32)

        qualities = [95, 90, 85, 75]
        diffs = []

        for q in qualities:
            temp_io = io.BytesIO()
            original.save(temp_io, 'JPEG', quality=q)
            temp_io.seek(0)
            compressed = Image.open(temp_io).convert('RGB')

            diff = np.abs(orig_arr - np.array(compressed, dtype=np.float32))
            # Convert to grayscale diff
            gray_diff = np.mean(diff, axis=2)

            max_val = gray_diff.max()
            if max_val > 0:
                gray_diff = gray_diff / max_val * 255.0

            diffs.append(gray_diff)

        avg_diff = np.mean(diffs, axis=0)

        # --- Core ELA metrics ---
        mean_ela = float(np.mean(avg_diff))
        std_ela = float(np.std(avg_diff))
        max_ela = float(np.max(avg_diff))
        uniformity = std_ela / (mean_ela + 1e-6)

        # --- Block variance (8x8 JPEG block aligned) ---
        h, w = avg_diff.shape
        block_size = 8
        block_means = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = avg_diff[y:y + block_size, x:x + block_size]
                block_means.append(np.mean(block))

        block_variance = float(np.var(block_means)) if block_means else 0.0

        # --- Quadrant uniformity analysis ---
        # AI images tend to have very uniform ELA across quadrants
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            avg_diff[:mid_h, :mid_w],
            avg_diff[:mid_h, mid_w:],
            avg_diff[mid_h:, :mid_w],
            avg_diff[mid_h:, mid_w:]
        ]
        quadrant_means = [float(np.mean(q)) for q in quadrants]
        quadrant_stds = [float(np.std(q)) for q in quadrants]
        quadrant_cv = float(np.std(quadrant_means) / (np.mean(quadrant_means) + 1e-6))

        # --- ELA Entropy ---
        # Histogram of ELA values → entropy. AI images tend to have lower ELA entropy.
        hist, _ = np.histogram(avg_diff.ravel(), bins=64, range=(0, 256))
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        ela_entropy = float(-np.sum(probs * np.log2(probs)))

        # --- Ghost JPEG detection (double compression) ---
        # Re-save at quality 90, compare with q75 diff — ghost indicates prior compression
        ghost_score = 0
        if len(diffs) >= 4:
            diff_high = diffs[0]  # q95
            diff_low = diffs[3]   # q75
            ratio = np.mean(diff_low) / (np.mean(diff_high) + 1e-6)
            if ratio < 1.5:
                ghost_score = 15  # Suspiciously similar → possible prior JPEG compression

        # --- Scoring Logic ---
        score = 50

        # Very uniform ELA = suspicious (AI typically has uniform compression)
        if std_ela < 5 and mean_ela < 10:
            score += 25
        elif std_ela < 10 and mean_ela < 15:
            score += 15
        elif std_ela > 40:
            score -= 15

        # Low block variance = suspicious (AI has uniform texture)
        if block_variance < 10:
            score += 12
        elif block_variance > 80:
            score -= 12

        # Quadrant uniformity
        if quadrant_cv < 0.15:
            score += 10  # Very uniform across quadrants → AI likely
        elif quadrant_cv > 0.5:
            score -= 8   # Variable across quadrants → more natural

        # ELA entropy
        if ela_entropy < 3.5:
            score += 8
        elif ela_entropy > 5.5:
            score -= 5

        # Ghost JPEG
        score += ghost_score

        # --- Visualization ---
        ela_img = colorize_heatmap(avg_diff.astype(np.uint8))

        buffered = io.BytesIO()
        ela_img.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            'score': min(max(int(score), 0), 100),
            'details': {
                'metrics': {
                    'mean_ela': round(mean_ela, 2),
                    'std_ela': round(std_ela, 2),
                    'max_ela': round(max_ela, 2),
                    'uniformity': round(uniformity, 4),
                    'block_variance': round(block_variance, 2),
                    'quadrant_cv': round(quadrant_cv, 4),
                    'ela_entropy': round(ela_entropy, 3),
                    'ghost_jpeg_indicator': bool(ghost_score > 0)
                }
            },
            'visualization': img_str
        }

    except Exception as e:
        return {
            'score': 50,
            'details': {'metrics': {}},
            'visualization': ''
        }
