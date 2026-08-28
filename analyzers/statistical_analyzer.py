import numpy as np
from scipy import stats
from PIL import Image


def analyze_statistical(image_path: str) -> dict:
    """
    Statistical analysis of image pixel distributions, color physics, and texture.
    Features evaluated:
    - Scale-invariant Benford's Law adherence on gradient magnitudes
    - Chromatic saturation distribution & non-physical color dynamics
    - Gray-Level Co-occurrence Matrix (GLCM) texture homogeneity and contrast
    - Shannon entropy across color channels
    Returns: {'score': int, 'details': {...}, 'histogram_data': {'r':[], 'g':[], 'b':[]}}
    """
    try:
        img_rgb = Image.open(image_path).convert('RGB')
        arr_rgb = np.array(img_rgb)
        h, w, _ = arr_rgb.shape

        metrics = {}
        down_hists = {'r': [], 'g': [], 'b': []}

        total_entropy = 0
        total_kurtosis = 0

        # --- 1. Per-Channel Statistics & Histograms ---
        for c, color in enumerate(['r', 'g', 'b']):
            channel = arr_rgb[:, :, c]

            # Histograms for UI visualization (32 bins)
            down_hist, _ = np.histogram(channel.ravel(), bins=32, range=(0, 256))
            down_hists[color] = down_hist.tolist()

            # Shannon Entropy
            hist, _ = np.histogram(channel.ravel(), bins=256, range=(0, 256))
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            entropy = float(-np.sum(probs * np.log2(probs)))

            kurt = float(stats.kurtosis(channel.ravel()))
            skew = float(stats.skew(channel.ravel()))

            total_entropy += entropy
            total_kurtosis += kurt

            metrics[f'{color}_entropy'] = round(entropy, 4)
            metrics[f'{color}_mean'] = round(float(np.mean(channel)), 2)
            metrics[f'{color}_std'] = round(float(np.std(channel)), 2)

        avg_entropy = float(total_entropy / 3.0)
        avg_kurt = float(total_kurtosis / 3.0)
        metrics['avg_entropy'] = round(avg_entropy, 4)

        # --- 2. Scale-Invariant Benford's Law on Image Gradients ---
        # Convert to grayscale
        gray = 0.299 * arr_rgb[:, :, 0] + 0.587 * arr_rgb[:, :, 1] + 0.114 * arr_rgb[:, :, 2]
        dy, dx = np.gradient(gray.astype(np.float64))
        grad_mag = np.hypot(dx, dy)

        # First digit analysis with scale invariance (Total Absolute Deviation)
        flat_grad = grad_mag.ravel()
        non_zero_grad = flat_grad[flat_grad >= 1.0]

        if len(non_zero_grad) > 500:
            log10_vals = np.floor(np.log10(non_zero_grad))
            first_digits = np.floor(non_zero_grad / (10.0 ** log10_vals)).astype(int)
            first_digits = np.clip(first_digits, 1, 9)

            actual_counts = np.bincount(first_digits, minlength=10)[1:10]
            total_valid = np.sum(actual_counts)

            actual_probs = actual_counts / total_valid
            expected_probs = np.log10(1.0 + 1.0 / np.arange(1, 10))

            # Scale-invariant Total Absolute Deviation (TAD)
            benford_dev = float(np.sum(np.abs(actual_probs - expected_probs)))
            # Correlation with Benford curve
            benford_corr = float(np.corrcoef(actual_probs, expected_probs)[0, 1])
        else:
            actual_probs = np.zeros(9)
            expected_probs = np.log10(1.0 + 1.0 / np.arange(1, 10))
            benford_dev = 0.04
            benford_corr = 0.95

        metrics['benford_deviation'] = round(benford_dev, 4)
        metrics['benford_correlation'] = round(benford_corr, 4)

        # --- 3. Color Saturation & Chromatic Physics (HSV analysis) ---
        # Modern diffusion generators (Midjourney v6, SDXL, Flux) produce distinctive
        # non-physical hyper-saturation in highlights and midtones
        max_c = np.max(arr_rgb, axis=2).astype(np.float32)
        min_c = np.min(arr_rgb, axis=2).astype(np.float32)
        delta = max_c - min_c
        saturation = np.where(max_c > 0, delta / (max_c + 1e-6), 0.0)

        sat_mean = float(np.mean(saturation))
        sat_p90 = float(np.percentile(saturation, 90))
        sat_std = float(np.std(saturation))

        metrics['sat_mean'] = round(sat_mean, 4)
        metrics['sat_p90'] = round(sat_p90, 4)
        metrics['sat_std'] = round(sat_std, 4)

        # --- 4. GLCM Texture & Homogeneity Analysis ---
        # Quantize to 32 levels for rapid robust GLCM computation
        gray_uint8 = np.clip((gray / (gray.max() + 1e-6) * 31), 0, 31).astype(np.uint8)
        num_levels = 32

        glcm_h = np.zeros((num_levels, num_levels), dtype=np.int64)
        left = gray_uint8[:, :-1].ravel()
        right = gray_uint8[:, 1:].ravel()
        np.add.at(glcm_h, (left, right), 1)

        glcm_sum = glcm_h.sum()
        if glcm_sum > 0:
            glcm_norm = glcm_h / glcm_sum
            i_idx, j_idx = np.meshgrid(np.arange(num_levels), np.arange(num_levels), indexing='ij')

            glcm_contrast = float(np.sum(glcm_norm * (i_idx - j_idx) ** 2))
            glcm_homogeneity = float(np.sum(glcm_norm / (1.0 + np.abs(i_idx - j_idx))))
            glcm_energy = float(np.sum(glcm_norm ** 2))
        else:
            glcm_contrast = 15.0
            glcm_homogeneity = 0.5
            glcm_energy = 0.005

        metrics['glcm_contrast'] = round(glcm_contrast, 4)
        metrics['glcm_homogeneity'] = round(glcm_homogeneity, 4)
        metrics['glcm_energy'] = round(glcm_energy, 6)

        # --- 5. Symmetrical Calibrated Scoring Model ---
        score = 50.0

        # Benford's Law adherence: Natural gradients adhere closely (benford_dev < 0.04, corr > 0.98)
        if benford_dev < 0.030 and benford_corr > 0.985:
            score -= 16  # Authentic natural gradient physics
        elif benford_dev < 0.050 and benford_corr > 0.96:
            score -= 8
        elif benford_dev > 0.12 or benford_corr < 0.90:
            score += 16  # Synthetic gradient disruption
        elif benford_dev > 0.08:
            score += 8

        # Saturation Dynamics: AI models often exhibit extreme cinematic saturation profiles
        if sat_p90 > 0.85 and sat_mean > 0.55:
            score += 14  # Non-physical hyper-saturation profile
        elif sat_p90 < 0.65 and sat_mean < 0.35:
            score -= 10  # Natural optical sensor color dynamics

        # GLCM Homogeneity vs Contrast: Overly smooth/plastic AI textures
        if glcm_homogeneity > 0.75 and glcm_energy > 0.015:
            score += 14  # Overly smooth synthetic texture
        elif glcm_contrast > 25.0:
            score -= 8   # Authentic fine natural surface texture

        # Entropy balance
        if avg_entropy > 7.5:
            score -= 6   # Rich optical information depth
        elif avg_entropy < 6.0:
            score += 10  # Quantized/reduced synthetic dynamic range

        final_score = int(round(min(max(score, 0), 100)))

        return {
            'score': final_score,
            'details': {
                'metrics': metrics,
                'benford_actual': actual_probs.tolist(),
                'benford_expected': expected_probs.tolist()
            },
            'histogram_data': down_hists
        }

    except Exception as e:
        return {
            'score': 50,
            'details': {'metrics': {}, 'benford_actual': [], 'benford_expected': [], 'error': str(e)},
            'histogram_data': {'r': [], 'g': [], 'b': []}
        }
