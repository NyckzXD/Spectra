import io
import base64
import numpy as np
from scipy import stats, ndimage
from PIL import Image


def analyze_noise(image_path: str) -> dict:
    """
    Analyzes physical sensor noise patterns vs synthetic diffusion residuals.
    Features evaluated:
    - Poisson-Gaussian photon noise model (variance vs intensity)
    - Bayer CFA demosaicing cross-channel noise correlation
    - Spatial noise distribution, kurtosis, and block consistency
    Returns: {'score': int, 'details': {'metrics': {...}}, 'visualization': 'base64string'}
    """
    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img, dtype=np.float32)
        h, w, _ = arr.shape

        # Downscale for performance if image is huge (preserve forensic characteristics)
        if max(h, w) > 2048:
            scale = 2048 / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            img_scaled = img.resize(new_size, Image.Resampling.BILINEAR)
            arr = np.array(img_scaled, dtype=np.float32)
            h, w, _ = arr.shape

        # --- Noise isolation via high-pass filtering (Wavelet/Laplacian approximation) ---
        smoothed = ndimage.gaussian_filter(arr, sigma=(1.5, 1.5, 0))
        noise = arr - smoothed

        # Luminance & Grayscale noise
        gray_arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        gray_noise = 0.299 * noise[:, :, 0] + 0.587 * noise[:, :, 1] + 0.114 * noise[:, :, 2]

        metrics = {}
        channel_stds = []

        for c, color in enumerate(['r', 'g', 'b']):
            ch_noise = noise[:, :, c]
            std = float(np.std(ch_noise))
            kurt = float(stats.kurtosis(ch_noise.ravel()))
            skew = float(stats.skew(ch_noise.ravel()))
            channel_stds.append(std)

            metrics[f'{color}_noise_std'] = round(std, 4)
            metrics[f'{color}_kurtosis'] = round(kurt, 4)
            metrics[f'{color}_skewness'] = round(skew, 4)

        avg_std = float(np.mean(channel_stds))
        metrics['avg_noise_std'] = round(avg_std, 4)

        # --- 1. Poisson-Gaussian Sensor Noise Test (Variance vs Local Luminance) ---
        # In real cameras, photon shot noise increases with brightness: Var(noise) = a * Intensity + b
        # AI images do not obey this physical law (uniform noise or inverse/flat relation)
        block_size = 16
        intensities = []
        noise_variances = []

        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block_img = gray_arr[y:y + block_size, x:x + block_size]
                block_noise = gray_noise[y:y + block_size, x:x + block_size]

                # Filter out pure uniform/clipping blocks (overexposed or pure black)
                mean_int = np.mean(block_img)
                if 15 < mean_int < 240:
                    # Robust noise variance (using Median Absolute Deviation to resist edges)
                    mad = np.median(np.abs(block_noise - np.median(block_noise)))
                    noise_var = (1.4826 * mad) ** 2
                    intensities.append(mean_int)
                    noise_variances.append(noise_var)

        poisson_corr = 0.0
        if len(intensities) > 30:
            intensities_arr = np.array(intensities)
            variances_arr = np.array(noise_variances)
            # Remove extreme outliers for correlation calculation
            p95 = np.percentile(variances_arr, 95)
            valid_mask = variances_arr < p95
            if np.sum(valid_mask) > 20:
                corr, _ = stats.spearmanr(intensities_arr[valid_mask], variances_arr[valid_mask])
                if not np.isnan(corr):
                    poisson_corr = float(corr)

        metrics['poisson_correlation'] = round(poisson_corr, 4)

        # --- 2. Inter-Channel Noise Correlation (Bayer CFA CFA Demosaicing) ---
        noise_r = noise[:, :, 0].ravel()
        noise_g = noise[:, :, 1].ravel()
        noise_b = noise[:, :, 2].ravel()

        sample_size = min(40000, len(noise_r))
        idx = np.random.choice(len(noise_r), sample_size, replace=False)

        rg_corr = float(np.corrcoef(noise_r[idx], noise_g[idx])[0, 1])
        rb_corr = float(np.corrcoef(noise_r[idx], noise_b[idx])[0, 1])
        gb_corr = float(np.corrcoef(noise_g[idx], noise_b[idx])[0, 1])
        avg_channel_corr = float((rg_corr + rb_corr + gb_corr) / 3.0)

        metrics['channel_correlation_rg'] = round(rg_corr, 4)
        metrics['channel_correlation_rb'] = round(rb_corr, 4)
        metrics['channel_correlation_gb'] = round(gb_corr, 4)
        metrics['avg_channel_correlation'] = round(avg_channel_corr, 4)

        # --- 3. Spatial Autocorrelation & High-Frequency Uniformity ---
        lag_h1 = gray_noise[:, :-1].ravel()[:sample_size]
        lag_h2 = gray_noise[:, 1:].ravel()[:sample_size]
        autocorr_h = float(np.corrcoef(lag_h1, lag_h2)[0, 1]) if len(lag_h1) > 10 else 0.0

        lag_v1 = gray_noise[:-1, :].ravel()[:sample_size]
        lag_v2 = gray_noise[1:, :].ravel()[:sample_size]
        autocorr_v = float(np.corrcoef(lag_v1, lag_v2)[0, 1]) if len(lag_v1) > 10 else 0.0

        avg_autocorr = float((autocorr_h + autocorr_v) / 2.0)
        metrics['avg_autocorrelation'] = round(avg_autocorr, 4)

        # --- 4. Symmetrical Calibrated Scoring Model ---
        score = 50.0

        # Poisson-Gaussian model: Real cameras show positive correlation between light and noise
        if poisson_corr > 0.30:
            score -= 22  # Strong real sensor signature
        elif poisson_corr > 0.15:
            score -= 12  # Moderate real camera indicator
        elif poisson_corr < -0.10:
            score += 18  # Non-physical noise behavior (AI signature)
        elif poisson_corr < 0.03:
            score += 8   # Neutral/Flat noise distribution

        # Channel Correlation: Real Bayer sensors correlate R, G, B noise between 0.35 and 0.85
        if 0.35 <= avg_channel_corr <= 0.85:
            score -= 12  # Real Bayer sensor demosaicing
        elif avg_channel_corr < 0.15:
            score += 15  # Synthetic independent channel synthesis
        elif avg_channel_corr > 0.96:
            score += 12  # Artificial exact monochrome noise injection

        # Noise Magnitude: Unnaturally zero noise vs realistic sensor grain
        if avg_std < 0.8:
            score += 16  # Overly pristine / smooth AI generation
        elif avg_std < 1.4:
            score += 8
        elif 2.5 <= avg_std <= 8.0:
            score -= 8   # Typical natural camera ISO noise

        # Spatial Autocorrelation: Diffusion artifacts often introduce spatial blur correlation
        if avg_autocorr > 0.45:
            score += 12  # Unnatural spatial blur / VAE residual
        elif avg_autocorr < 0.12:
            score -= 6   # True random sensor shot noise

        # Clamp score to [0, 100]
        final_score = int(round(min(max(score, 0), 100)))

        # --- Visualization: High-contrast noise map ---
        noise_norm = (gray_noise - gray_noise.min()) / (gray_noise.max() - gray_noise.min() + 1e-6)
        noise_norm = np.clip((noise_norm - 0.5) * 4.0 + 0.5, 0, 1)

        vis_img = Image.fromarray((noise_norm * 255).astype(np.uint8))
        buffered = io.BytesIO()
        vis_img.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            'score': final_score,
            'details': {'metrics': metrics},
            'visualization': img_str
        }

    except Exception as e:
        return {
            'score': 50,
            'details': {'metrics': {}, 'error': str(e)},
            'visualization': ''
        }
