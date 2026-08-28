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
    - Natural optical image power-law decay: Amplitude A(f) ~ 1 / f^alpha (alpha ~ 1.0)
    - Goodness of fit (R^2) of the 1/f falloff curve
    - High-frequency spectral flatness and residual noise floor
    - Anomalous 2D harmonic peaks (GAN/Diffusion upsampling fingerprints)
    - Azimuthal directional symmetry vs natural scene orientation
    Returns: {'score': int, 'details': {'metrics': {...}}, 'visualization': 'base64string'}
    """
    try:
        img_rgb = Image.open(image_path).convert('RGB')
        arr_rgb = np.array(img_rgb, dtype=np.float64)
        h, w, _ = arr_rgb.shape

        # Downsample if image is overly large while preserving spectral structure
        if max(h, w) > 1536:
            scale = 1536 / max(h, w)
            img_rgb = img_rgb.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            arr_rgb = np.array(img_rgb, dtype=np.float64)
            h, w, _ = arr_rgb.shape

        # Grayscale luminance
        gray = 0.299 * arr_rgb[:, :, 0] + 0.587 * arr_rgb[:, :, 1] + 0.114 * arr_rgb[:, :, 2]

        # Apply Hanning window to reduce edge boundary leakage in FFT
        han_y = np.hanning(h)
        han_x = np.hanning(w)
        window = np.outer(han_y, han_x)
        windowed_gray = (gray - np.mean(gray)) * window

        # --- 1. 2D FFT & Linear Amplitude Spectrum ---
        f = np.fft.fft2(windowed_gray)
        fshift = np.fft.fftshift(f)
        amp = np.abs(fshift)  # Linear amplitude

        # Radial profile calculation on linear amplitude
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.ogrid[0:h, 0:w]
        r = np.hypot(x_grid - cx, y_grid - cy).astype(int)

        max_r = min(cy, cx)
        if max_r < 16:
            max_r = 16

        tbin = np.bincount(r.ravel(), amp.ravel())
        nr = np.bincount(r.ravel())
        radial_amp = tbin / np.maximum(nr, 1)
        radial_amp = radial_amp[:max_r]

        # --- 2. Spectral Slope (Alpha) via Single-Log Linear Regression ---
        # Natural optical images follow A(f) ~ 1/f^alpha, so ln(A) = -alpha * ln(f) + C
        # Alpha is typically between 0.85 and 1.35 for authentic photography.
        valid_len = len(radial_amp)
        if valid_len > 12:
            # Frequency indices (avoid DC 0..3 and Nyquist boundary near max_r)
            f_start = 3
            f_end = int(max_r * 0.85)
            f_range = np.arange(f_start, f_end)

            freq_vals = f_range.astype(np.float64)
            amp_vals = radial_amp[f_start:f_end] + 1e-6

            log_f = np.log(freq_vals)
            log_amp = np.log(amp_vals)

            slope, intercept, r_value, p_value, std_err = stats.linregress(log_f, log_amp)
            alpha = float(-slope)
            r_squared = float(r_value ** 2)

            # --- 3. Anomalous Periodic Spikes Detection ---
            # Measure deviations from the power-law fit
            expected_log_amp = intercept + slope * log_f
            residuals = log_amp - expected_log_amp
            std_res = float(np.std(residuals))

            if std_res > 0:
                # Spikes are points significantly above the power-law regression
                peak_mask = residuals > 2.2 * std_res
                anomalous_peaks = int(np.sum(peak_mask))
                peak_strength = float(np.mean(residuals[peak_mask] / std_res)) if anomalous_peaks > 0 else 0.0
            else:
                anomalous_peaks = 0
                peak_strength = 0.0
        else:
            alpha = 1.0
            r_squared = 0.98
            anomalous_peaks = 0
            peak_strength = 0.0

        # --- 4. High-Frequency Spectral Flatness (Noise Floor Analysis) ---
        # AI diffusion images often exhibit an unnaturally flat, elevated high-frequency tail
        hf_start = int(max_r * 0.5)
        hf_end = int(max_r * 0.9)
        if hf_end > hf_start + 4:
            hf_band = radial_amp[hf_start:hf_end] + 1e-6
            # Spectral flatness = geometric mean / arithmetic mean (1.0 = pure white noise, 0.0 = tonal)
            geom_mean = float(np.exp(np.mean(np.log(hf_band))))
            arith_mean = float(np.mean(hf_band))
            hf_flatness = float(geom_mean / (arith_mean + 1e-6))
        else:
            hf_flatness = 0.5

        # --- 5. High-Frequency Energy Ratio (Luminance HF vs LF) ---
        mid_r = max_r // 2
        hf_mask = (r > mid_r) & (r <= max_r)
        lf_mask = (r > 3) & (r <= mid_r)

        hf_energy = np.mean(amp[hf_mask]) + 1e-6
        lf_energy = np.mean(amp[lf_mask]) + 1e-6
        hf_lf_ratio = float(hf_energy / lf_energy)

        # --- 6. Calibrated Symmetrical Scoring Model ---
        score = 50.0

        # Natural Power-Law Slope Alpha:
        # Optical real photos: 0.85 <= alpha <= 1.30 with high R^2 (> 0.94)
        if 0.85 <= alpha <= 1.30 and r_squared > 0.94:
            score -= 20  # Strong natural optical 1/f falloff
        elif 0.70 <= alpha <= 1.45 and r_squared > 0.90:
            score -= 10  # Moderate natural falloff
        elif alpha < 0.55 or alpha > 1.65:
            score += 18  # Synthetic non-physical spectral decay
        elif r_squared < 0.82:
            score += 14  # Fragmented non-natural spectrum

        # High-Frequency Flatness (AI diffusion noise floor):
        # AI images tend to have flatter high-frequency noise floor (> 0.80)
        # Natural optical falloff has lower flatness (< 0.65)
        if hf_flatness > 0.85:
            score += 14  # Flat synthetic noise floor (diffusion signature)
        elif hf_flatness < 0.60:
            score -= 10  # Natural smooth optical attenuation

        # Anomalous Harmonic Spikes (GAN / Upscaler periodic fingerprints):
        if anomalous_peaks >= 4:
            score += min(anomalous_peaks * 4, 20)
        elif anomalous_peaks == 0:
            score -= 6   # Clean continuous natural spectrum

        # Extreme High-to-Low frequency energy ratios:
        if hf_lf_ratio < 0.005:
            score += 10  # Overly blurred/smoothed synthetic image
        elif 0.015 <= hf_lf_ratio <= 0.15:
            score -= 6   # Natural balance of photographic detail

        final_score = int(round(min(max(score, 0), 100)))

        # --- Visualization: 2D Log-Magnitude Spectrum ---
        log_mag_vis = 20 * np.log(amp + 1)
        mag_min = log_mag_vis.min()
        mag_max = log_mag_vis.max()
        mag_norm = (log_mag_vis - mag_min) / (mag_max - mag_min + 1e-6)

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
                    'spectral_slope_alpha': round(alpha, 4),
                    'power_law_fit_r2': round(r_squared, 4),
                    'hf_spectral_flatness': round(hf_flatness, 4),
                    'hf_lf_energy_ratio': round(hf_lf_ratio, 4),
                    'anomalous_peaks': anomalous_peaks,
                    'peak_strength': round(peak_strength, 2),
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
