import io
import base64
import numpy as np
from scipy import stats
from PIL import Image


def vectorized_viridis(normalized):
    """Vectorized viridis-like colormap for spectral visualization."""
    r = np.clip(np.where(normalized < 0.5, normalized * 0.3,
                np.where(normalized < 0.75, 0.15 + (normalized - 0.5) * 2.4,
                         0.75 + (normalized - 0.75) * 1.0)), 0, 1)
    g = np.clip(np.where(normalized < 0.25, 0.05 + normalized * 1.2,
                np.where(normalized < 0.75, 0.35 + (normalized - 0.25) * 1.2,
                         0.95 - (normalized - 0.75) * 1.2)), 0, 1)
    b = np.clip(np.where(normalized < 0.5, 0.3 + normalized * 0.8,
                np.where(normalized < 0.75, 0.7 - (normalized - 0.5) * 2.0,
                         0.2 - (normalized - 0.75) * 0.8)), 0, 1)
    return r, g, b


def analyze_spectral(image_path: str) -> dict:
    """
    Analyzes frequency spectrum of the image using FFT.
    Detects GAN upsampling artifacts, anomalous spectral peaks, and radial symmetry.
    Returns: {'score': int, 'details': {'metrics': {...}}, 'visualization': 'base64string'}
    """
    try:
        img = Image.open(image_path).convert('L')
        arr = np.array(img, dtype=np.float64)

        # --- FFT ---
        f = np.fft.fft2(arr)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        # --- Radial profile ---
        h, w = arr.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[0:h, 0:w]
        r = np.hypot(x - cx, y - cy).astype(int)

        max_r = min(cy, cx)
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / np.maximum(nr, 1)

        # Trim to meaningful range
        radialprofile = radialprofile[:max_r]

        # --- Spectral slope (beta) via linear regression in log-log space ---
        valid_len = len(radialprofile)
        if valid_len > 10:
            x_log = np.log(np.arange(1, valid_len) + 1)
            y_log = np.log(radialprofile[1:] + 1)

            slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
            beta = -slope
            r_squared = r_value ** 2
        else:
            beta = 2.0
            r_squared = 0.0
            intercept = 0.0
            slope = 0.0
            x_log = np.array([])
            y_log = np.array([])

        # --- Anomalous peaks detection ---
        # Peaks that deviate significantly from the power-law fit
        peaks = 0
        peak_strength = 0.0
        if len(x_log) > 0 and len(y_log) > 0:
            expected_y = intercept + slope * x_log
            residuals = y_log - expected_y
            std_res = np.std(residuals)

            if std_res > 0:
                peak_mask = residuals > 2.0 * std_res
                peaks = int(np.sum(peak_mask))
                if peaks > 0:
                    peak_strength = float(np.mean(residuals[peak_mask] / std_res))

        # --- Spectral entropy ---
        norm_mag = magnitude_spectrum / (np.sum(magnitude_spectrum) + 1e-10)
        spectral_entropy = float(-np.sum(norm_mag * np.log2(norm_mag + 1e-10)))

        # --- Radial symmetry analysis ---
        # GANs tend to produce more radially symmetric spectra
        # Compute variance along angular direction for different radii
        angles = np.arctan2(y - cy, x - cx)
        radial_symmetry_scores = []
        for radius in [max_r // 4, max_r // 2, 3 * max_r // 4]:
            ring_mask = (r >= radius - 2) & (r <= radius + 2)
            ring_values = magnitude_spectrum[ring_mask]
            if len(ring_values) > 10:
                cv = np.std(ring_values) / (np.mean(ring_values) + 1e-6)
                radial_symmetry_scores.append(float(cv))

        avg_radial_cv = float(np.mean(radial_symmetry_scores)) if radial_symmetry_scores else 0.5

        # --- High frequency energy ratio ---
        # AI images often have less high-frequency content
        mid_r = max_r // 2
        low_freq_mask = r <= mid_r
        high_freq_mask = r > mid_r

        low_energy = np.mean(magnitude_spectrum[low_freq_mask])
        high_energy = np.mean(magnitude_spectrum[high_freq_mask])
        hf_ratio = high_energy / (low_energy + 1e-6)

        # --- Scoring ---
        score = 50

        # Beta: natural images ~1.5-2.5
        if beta < 1.0 or beta > 3.0:
            score += 18
        elif 1.5 <= beta <= 2.5:
            score -= 10

        # Anomalous peaks (GAN upsampling fingerprints)
        if peaks > 5:
            score += min(peaks * 3, 25)
        elif peaks > 2:
            score += peaks * 2

        # Low radial CV = high symmetry = more likely AI
        if avg_radial_cv < 0.15:
            score += 12
        elif avg_radial_cv > 0.4:
            score -= 8

        # Low high-frequency ratio → AI (often smoother)
        if hf_ratio < 0.3:
            score += 10
        elif hf_ratio > 0.6:
            score -= 8

        # --- Visualization (fully vectorized) ---
        mag_min = magnitude_spectrum.min()
        mag_max = magnitude_spectrum.max()
        mag_norm = (magnitude_spectrum - mag_min) / (mag_max - mag_min + 1e-6)

        r_ch, g_ch, b_ch = vectorized_viridis(mag_norm)
        color_spec = np.stack([
            (r_ch * 255).astype(np.uint8),
            (g_ch * 255).astype(np.uint8),
            (b_ch * 255).astype(np.uint8)
        ], axis=-1)

        spec_img = Image.fromarray(color_spec)
        buffered = io.BytesIO()
        spec_img.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            'score': min(max(int(score), 0), 100),
            'details': {
                'metrics': {
                    'spectral_slope_beta': round(float(beta), 4),
                    'r_squared': round(float(r_squared), 4),
                    'anomalous_peaks': int(peaks),
                    'peak_strength': round(peak_strength, 2),
                    'spectral_entropy': round(spectral_entropy, 3),
                    'radial_symmetry_cv': round(avg_radial_cv, 4),
                    'hf_energy_ratio': round(float(hf_ratio), 4),
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
