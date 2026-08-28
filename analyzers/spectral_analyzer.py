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
    Analyzes the 2D Fourier frequency spectrum of the image.
    Evaluates:
    - Natural image power-law slope (1/f^alpha)
    - Chrominance vs Luminance high-frequency decay (VAE latent compression fingerprint)
    - Azimuthal symmetry & periodic harmonic anomalies
    Returns: {'score': int, 'details': {'metrics': {...}}, 'visualization': 'base64string'}
    """
    try:
        img_rgb = Image.open(image_path).convert('RGB')
        arr_rgb = np.array(img_rgb, dtype=np.float64)
        h, w, _ = arr_rgb.shape

        # Downsample if image is overly large while preserving spectral characteristics
        if max(h, w) > 1536:
            scale = 1536 / max(h, w)
            img_rgb = img_rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            arr_rgb = np.array(img_rgb, dtype=np.float64)
            h, w, _ = arr_rgb.shape

        # Convert to YCbCr to evaluate luminance and chrominance spectrums
        # Y  =  0.2990*R + 0.5870*G + 0.1140*B
        # Cb = -0.1687*R - 0.3313*G + 0.5000*B + 128
        # Cr =  0.5000*R - 0.4187*G - 0.0813*B + 128
        y_channel = 0.299 * arr_rgb[:, :, 0] + 0.587 * arr_rgb[:, :, 1] + 0.114 * arr_rgb[:, :, 2]
        cb_channel = -0.1687 * arr_rgb[:, :, 0] - 0.3313 * arr_rgb[:, :, 1] + 0.5 * arr_rgb[:, :, 2]
        cr_channel = 0.5 * arr_rgb[:, :, 0] - 0.4187 * arr_rgb[:, :, 1] - 0.0813 * arr_rgb[:, :, 2]

        # --- 1. FFT on Luminance ---
        f_y = np.fft.fft2(y_channel)
        fshift_y = np.fft.fftshift(f_y)
        mag_y = np.abs(fshift_y)
        magnitude_spectrum = 20 * np.log(mag_y + 1)

        # Radial profile calculation
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.ogrid[0:h, 0:w]
        r = np.hypot(x_grid - cx, y_grid - cy).astype(int)

        max_r = min(cy, cx)
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / np.maximum(nr, 1)
        radial_profile = radial_profile[:max_r]

        # --- 2. Spectral Slope (Beta) & Power-Law Fit ---
        valid_len = len(radial_profile)
        if valid_len > 12:
            # Skip DC and extreme low freqs (0..3) to avoid windowing bias
            x_log = np.log(np.arange(4, valid_len) + 1)
            y_log = np.log(radial_profile[4:] + 1)

            slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
            beta = float(-slope)
            r_squared = float(r_value ** 2)

            # Check for high-frequency anomalous peaks
            expected_y = intercept + slope * x_log
            residuals = y_log - expected_y
            std_res = np.std(residuals)
            peak_mask = residuals > 2.2 * std_res if std_res > 0 else np.zeros_like(residuals, dtype=bool)
            peaks = int(np.sum(peak_mask))
            peak_strength = float(np.mean(residuals[peak_mask] / std_res)) if peaks > 0 else 0.0
        else:
            beta = 2.0
            r_squared = 0.95
            peaks = 0
            peak_strength = 0.0

        # --- 3. VAE Latent Chrominance High-Frequency Decay ---
        # Latent diffusion decoders (SD/Flux/Midjourney) attenuate Cb/Cr high frequencies noticeably
        f_cb = np.fft.fftshift(np.fft.fft2(cb_channel))
        f_cr = np.fft.fftshift(np.fft.fft2(cr_channel))
        mag_chroma = (np.abs(f_cb) + np.abs(f_cr)) / 2.0

        mid_r = max_r // 2
        high_freq_mask = (r > mid_r) & (r <= max_r)
        low_freq_mask = (r > 4) & (r <= mid_r)

        hf_y = np.mean(mag_y[high_freq_mask]) + 1e-6
        lf_y = np.mean(mag_y[low_freq_mask]) + 1e-6
        y_hf_ratio = hf_y / lf_y

        hf_chroma = np.mean(mag_chroma[high_freq_mask]) + 1e-6
        lf_chroma = np.mean(mag_chroma[low_freq_mask]) + 1e-6
        chroma_hf_ratio = hf_chroma / lf_chroma

        # Ratio of luminance HF preservation vs chroma HF preservation
        vae_decay_ratio = float(y_hf_ratio / (chroma_hf_ratio + 1e-6))

        # --- 4. Azimuthal Radial Symmetry ---
        radial_symmetry_scores = []
        for radius in [max_r // 4, max_r // 2, 3 * max_r // 4]:
            ring_mask = (r >= radius - 2) & (r <= radius + 2)
            ring_values = magnitude_spectrum[ring_mask]
            if len(ring_values) > 10:
                cv = np.std(ring_values) / (np.mean(ring_values) + 1e-6)
                radial_symmetry_scores.append(float(cv))

        avg_radial_cv = float(np.mean(radial_symmetry_scores)) if radial_symmetry_scores else 0.4

        # --- 5. Calibrated Symmetrical Scoring Model ---
        score = 50.0

        # Power law slope: Natural images strictly adhere to 1.7 <= beta <= 2.3 with high R^2
        if 1.7 <= beta <= 2.3 and r_squared > 0.95:
            score -= 16  # Strong natural optical power-law adherence
        elif 1.5 <= beta <= 2.5 and r_squared > 0.90:
            score -= 8   # Moderate natural adherence
        elif beta < 1.3 or beta > 2.8:
            score += 18  # Synthetic non-physical spectral roll-off
        elif r_squared < 0.85:
            score += 12  # Fragmented non-natural spectrum

        # VAE Chrominance High-Frequency Dropout: AI Diffusion fingerprint
        if vae_decay_ratio > 3.2:
            score += 16  # Distinct VAE latent chrominance smoothing
        elif vae_decay_ratio > 2.2:
            score += 8
        elif 0.7 <= vae_decay_ratio <= 1.8:
            score -= 10  # Natural optical sensor chrominance-luminance balance

        # Anomalous high-frequency spikes (GAN / upscaler harmonics)
        if peaks >= 4:
            score += min(peaks * 4, 20)

        # Azimuthal Symmetry: Unnaturally isotropic spectra in synthetic images
        if avg_radial_cv < 0.12:
            score += 12  # Overly isotropic synthetic generation
        elif avg_radial_cv > 0.35:
            score -= 8   # Natural directional structures (edges, horizons, shadows)

        final_score = int(round(min(max(score, 0), 100)))

        # --- Visualization (Vectorized 2D Spectrum) ---
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
            'score': final_score,
            'details': {
                'metrics': {
                    'spectral_slope_beta': round(beta, 4),
                    'power_law_fit_r2': round(r_squared, 4),
                    'vae_chroma_decay_ratio': round(vae_decay_ratio, 4),
                    'anomalous_peaks': peaks,
                    'peak_strength': round(peak_strength, 2),
                    'radial_symmetry_cv': round(avg_radial_cv, 4),
                    'hf_energy_ratio': round(float(y_hf_ratio), 4),
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
