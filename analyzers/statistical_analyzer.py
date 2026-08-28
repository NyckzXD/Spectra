import numpy as np
from scipy import stats, ndimage
from PIL import Image


def analyze_statistical(image_path: str) -> dict:
    """
    Statistical analysis of image pixels.
    Includes Benford's Law on gradients, simplified GLCM, pixel distribution smoothness.
    Returns: {'score': int, 'details': {...}, 'histogram_data': {'r':[], 'g':[], 'b':[]}}
    """
    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img)

        metrics = {}
        down_hists = {'r': [], 'g': [], 'b': []}

        total_entropy = 0
        total_kurtosis = 0

        for c, color in enumerate(['r', 'g', 'b']):
            channel = arr[:, :, c]

            # --- Histograms ---
            hist, _ = np.histogram(channel.ravel(), bins=256, range=(0, 256))
            down_hist, _ = np.histogram(channel.ravel(), bins=32, range=(0, 256))
            down_hists[color] = down_hist.tolist()

            # --- Shannon Entropy ---
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]
            entropy = float(-np.sum(probs * np.log2(probs)))

            # --- Kurtosis & Skewness ---
            kurt = float(stats.kurtosis(channel.ravel()))
            skew = float(stats.skew(channel.ravel()))

            total_entropy += entropy
            total_kurtosis += kurt

            metrics[f'{color}_entropy'] = round(entropy, 4)
            metrics[f'{color}_mean'] = round(float(np.mean(channel)), 2)
            metrics[f'{color}_std'] = round(float(np.std(channel)), 2)
            metrics[f'{color}_kurtosis'] = round(kurt, 4)
            metrics[f'{color}_skewness'] = round(skew, 4)

        avg_entropy = total_entropy / 3
        avg_kurt = total_kurtosis / 3

        # --- Benford's Law on gradient magnitudes (more informative than raw pixel values) ---
        gray = np.mean(arr.astype(np.float64), axis=2)
        dy, dx = np.gradient(gray)
        grad_mag = np.hypot(dx, dy)

        # Extract first digits from gradient magnitudes
        flat_grad = grad_mag.ravel()
        non_zero_grad = flat_grad[flat_grad >= 1.0]  # Only values >= 1

        if len(non_zero_grad) > 100:
            # Vectorized first digit extraction
            log10_vals = np.floor(np.log10(non_zero_grad))
            first_digits = np.floor(non_zero_grad / (10.0 ** log10_vals)).astype(int)
            first_digits = np.clip(first_digits, 1, 9)

            actual_counts = np.bincount(first_digits, minlength=10)[1:10]
            total_valid = np.sum(actual_counts)

            if total_valid > 0:
                actual_probs = actual_counts / total_valid
                expected_probs = np.log10(1 + 1 / np.arange(1, 10))
                expected_counts = expected_probs * total_valid

                chi_sq, chi_p = stats.chisquare(actual_counts, expected_counts)
                benford_deviation = float(np.mean(np.abs(actual_probs - expected_probs)))
            else:
                chi_sq = 0
                chi_p = 1.0
                actual_probs = np.zeros(9)
                expected_probs = np.log10(1 + 1 / np.arange(1, 10))
                benford_deviation = 0.0
        else:
            chi_sq = 0
            chi_p = 1.0
            actual_probs = np.zeros(9)
            expected_probs = np.log10(1 + 1 / np.arange(1, 10))
            benford_deviation = 0.0

        metrics['benford_chi_sq'] = round(float(chi_sq), 2)
        metrics['benford_p_value'] = round(float(chi_p), 6)
        metrics['benford_deviation'] = round(benford_deviation, 6)

        # --- Gradient statistics ---
        metrics['grad_mean'] = round(float(np.mean(grad_mag)), 4)
        metrics['grad_std'] = round(float(np.std(grad_mag)), 4)
        grad_ratio = float(np.std(grad_mag) / (np.mean(grad_mag) + 1e-6))
        metrics['grad_cv'] = round(grad_ratio, 4)

        # --- Simplified GLCM (Co-occurrence matrix) ---
        # Compute for grayscale, quantized to 32 levels for efficiency
        gray_uint8 = (gray / gray.max() * 31).astype(np.uint8)
        h, w = gray_uint8.shape
        num_levels = 32

        # Horizontal co-occurrence
        glcm_h = np.zeros((num_levels, num_levels), dtype=np.int64)
        left = gray_uint8[:, :-1].ravel()
        right = gray_uint8[:, 1:].ravel()
        np.add.at(glcm_h, (left, right), 1)

        # Normalize
        glcm_sum = glcm_h.sum()
        if glcm_sum > 0:
            glcm_norm = glcm_h / glcm_sum

            # GLCM features
            i_idx, j_idx = np.meshgrid(np.arange(num_levels), np.arange(num_levels), indexing='ij')

            # Contrast
            glcm_contrast = float(np.sum(glcm_norm * (i_idx - j_idx) ** 2))

            # Homogeneity (Inverse Difference Moment)
            glcm_homogeneity = float(np.sum(glcm_norm / (1 + np.abs(i_idx - j_idx))))

            # Energy (Angular Second Moment)
            glcm_energy = float(np.sum(glcm_norm ** 2))

            # Correlation
            mu_i = np.sum(i_idx * glcm_norm)
            mu_j = np.sum(j_idx * glcm_norm)
            sigma_i = np.sqrt(np.sum(glcm_norm * (i_idx - mu_i) ** 2))
            sigma_j = np.sqrt(np.sum(glcm_norm * (j_idx - mu_j) ** 2))
            if sigma_i > 0 and sigma_j > 0:
                glcm_correlation = float(np.sum(glcm_norm * (i_idx - mu_i) * (j_idx - mu_j)) / (sigma_i * sigma_j))
            else:
                glcm_correlation = 0.0
        else:
            glcm_contrast = 0.0
            glcm_homogeneity = 0.0
            glcm_energy = 0.0
            glcm_correlation = 0.0

        metrics['glcm_contrast'] = round(glcm_contrast, 4)
        metrics['glcm_homogeneity'] = round(glcm_homogeneity, 4)
        metrics['glcm_energy'] = round(glcm_energy, 6)
        metrics['glcm_correlation'] = round(glcm_correlation, 4)

        # --- Pixel distribution smoothness ---
        # AI generators tend to produce smoother histograms
        full_hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        hist_gradient = np.diff(full_hist.astype(np.float64))
        hist_smoothness = float(np.std(hist_gradient) / (np.mean(np.abs(hist_gradient)) + 1e-6))
        metrics['histogram_smoothness'] = round(hist_smoothness, 4)

        # --- Scoring ---
        score = 50

        # Entropy: very low = likely synthetic patterns
        if avg_entropy < 6.5:
            score += 15
        elif avg_entropy > 7.8:
            score -= 10

        # Benford: high chi-square = deviates from expected (synthetic)
        if chi_sq > 200:
            score += 12
        elif chi_sq > 100:
            score += 8
        elif chi_sq < 20:
            score -= 5

        # Kurtosis: very negative = platykurtic (flat distribution, can indicate AI)
        if avg_kurt < -1.0:
            score += 8

        # GLCM: high homogeneity + high energy = overly smooth texture (AI)
        if glcm_homogeneity > 0.7 and glcm_energy > 0.01:
            score += 10
        elif glcm_contrast > 50:
            score -= 8

        # Histogram smoothness: AI generates smoother histograms
        if hist_smoothness < 1.5:
            score += 8
        elif hist_smoothness > 3.0:
            score -= 5

        # Gradient CV: low variance-to-mean ratio in gradients → AI (uniform sharpness)
        if grad_ratio < 0.8:
            score += 8
        elif grad_ratio > 1.5:
            score -= 5

        return {
            'score': min(max(int(score), 0), 100),
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
            'details': {'metrics': {}, 'benford_actual': [], 'benford_expected': []},
            'histogram_data': {'r': [], 'g': [], 'b': []}
        }
