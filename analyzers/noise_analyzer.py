import io
import base64
import numpy as np
from scipy import stats, ndimage
from PIL import Image


def analyze_noise(image_path: str) -> dict:
    """
    Analyzes noise patterns in the image.
    Detects PRNU-like patterns, noise distribution anomalies, and spatial correlation.
    Returns: {'score': int, 'details': {'metrics': {...}}, 'visualization': 'base64string'}
    """
    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img, dtype=np.float32)

        # --- Noise isolation (Gaussian filtering) ---
        smoothed = ndimage.gaussian_filter(arr, sigma=(2.0, 2.0, 0))
        noise = arr - smoothed

        metrics = {}
        channel_stats = []
        block_cvs = []

        for c, color in enumerate(['r', 'g', 'b']):
            channel_noise = noise[:, :, c]
            std = float(np.std(channel_noise))
            mean_noise = float(np.mean(channel_noise))
            kurt = float(stats.kurtosis(channel_noise.ravel()))
            skew = float(stats.skew(channel_noise.ravel()))

            channel_stats.append({
                'std': std, 'kurt': kurt, 'skew': skew, 'mean': mean_noise
            })

            metrics[f'{color}_std'] = round(std, 4)
            metrics[f'{color}_kurtosis'] = round(kurt, 4)
            metrics[f'{color}_skewness'] = round(skew, 4)

            # Block-based noise consistency
            h, w = channel_noise.shape
            block_size = 8
            block_stds = []
            for y in range(0, h - block_size + 1, block_size):
                for x in range(0, w - block_size + 1, block_size):
                    block = channel_noise[y:y + block_size, x:x + block_size]
                    block_stds.append(np.std(block))

            if block_stds:
                cv = np.std(block_stds) / (np.mean(block_stds) + 1e-6)
                block_cvs.append(float(cv))

        avg_std = np.mean([s['std'] for s in channel_stats])
        avg_kurt = np.mean([s['kurt'] for s in channel_stats])
        avg_cv = float(np.mean(block_cvs)) if block_cvs else 0.0

        # --- Inter-channel noise consistency ---
        # Real cameras have correlated noise across channels (same sensor)
        # AI generators produce noise independently per channel
        noise_r = noise[:, :, 0].ravel()
        noise_g = noise[:, :, 1].ravel()
        noise_b = noise[:, :, 2].ravel()

        # Sample for performance (correlation on full image is expensive)
        sample_size = min(50000, len(noise_r))
        idx = np.random.choice(len(noise_r), sample_size, replace=False)

        rg_corr = float(np.corrcoef(noise_r[idx], noise_g[idx])[0, 1])
        rb_corr = float(np.corrcoef(noise_r[idx], noise_b[idx])[0, 1])
        gb_corr = float(np.corrcoef(noise_g[idx], noise_b[idx])[0, 1])
        avg_channel_corr = (rg_corr + rb_corr + gb_corr) / 3.0

        metrics['channel_correlation_rg'] = round(rg_corr, 4)
        metrics['channel_correlation_rb'] = round(rb_corr, 4)
        metrics['channel_correlation_gb'] = round(gb_corr, 4)
        metrics['avg_channel_correlation'] = round(avg_channel_corr, 4)

        # --- Spatial autocorrelation ---
        gray_noise = np.mean(noise, axis=2)
        # Horizontal lag-1
        lag_h1 = gray_noise[:, :-1]
        lag_h2 = gray_noise[:, 1:]
        autocorr_h = float(np.corrcoef(lag_h1.ravel()[:sample_size],
                                       lag_h2.ravel()[:sample_size])[0, 1])
        # Vertical lag-1
        lag_v1 = gray_noise[:-1, :]
        lag_v2 = gray_noise[1:, :]
        autocorr_v = float(np.corrcoef(lag_v1.ravel()[:sample_size],
                                       lag_v2.ravel()[:sample_size])[0, 1])
        avg_autocorr = (autocorr_h + autocorr_v) / 2.0

        metrics['autocorrelation_h'] = round(autocorr_h, 4)
        metrics['autocorrelation_v'] = round(autocorr_v, 4)
        metrics['avg_autocorrelation'] = round(avg_autocorr, 4)
        metrics['block_std_cv'] = round(avg_cv, 4)

        # --- Noise distribution test ---
        # Real sensor noise should be approximately Gaussian
        # Use Shapiro-Wilk on a small sample
        noise_sample = gray_noise.ravel()
        sample_for_test = np.random.choice(noise_sample, min(5000, len(noise_sample)), replace=False)
        try:
            _, p_value = stats.normaltest(sample_for_test)
            is_gaussian = p_value > 0.05
        except Exception:
            p_value = 0.5
            is_gaussian = True

        metrics['noise_gaussian_p'] = round(float(p_value), 6)
        metrics['noise_is_gaussian'] = is_gaussian

        # --- PRNU-like analysis (simplified) ---
        # Real cameras have fixed-pattern noise. We check if noise has spatial structure.
        # Divide image into overlapping patches and check noise pattern consistency
        patch_size = min(64, h // 4, w // 4)
        if patch_size >= 16:
            patch_correlations = []
            patches = []
            for y in range(0, h - patch_size + 1, patch_size):
                for x in range(0, w - patch_size + 1, patch_size):
                    patch = gray_noise[y:y + patch_size, x:x + patch_size]
                    patches.append(patch.ravel())

            # Check correlation between distant patches (PRNU would create correlation)
            if len(patches) >= 4:
                for i in range(min(10, len(patches) - 1)):
                    j = min(i + len(patches) // 2, len(patches) - 1)
                    if i != j:
                        corr = np.corrcoef(patches[i], patches[j])[0, 1]
                        if not np.isnan(corr):
                            patch_correlations.append(float(corr))

            prnu_indicator = float(np.mean(patch_correlations)) if patch_correlations else 0.0
        else:
            prnu_indicator = 0.0

        metrics['prnu_indicator'] = round(prnu_indicator, 4)

        # --- Scoring ---
        score = 50

        # Very low noise = suspicious (AI often has unnaturally clean images)
        if avg_std < 1.5:
            score += 20
        elif avg_std < 3.0:
            score += 10
        elif avg_std > 10:
            score -= 10

        # Low block CV = uniform noise = suspicious for AI
        if avg_cv < 0.3:
            score += 12
        elif avg_cv > 0.6:
            score -= 8

        # High spatial autocorrelation = suspicious (AI noise is often spatially correlated)
        if avg_autocorr > 0.3:
            score += 10
        elif avg_autocorr < 0.05:
            score -= 5

        # Low inter-channel correlation = suspicious for AI
        if avg_channel_corr < 0.1:
            score += 8
        elif avg_channel_corr > 0.5:
            score -= 8  # High correlation = sensor noise (real)

        # Non-Gaussian noise
        if not is_gaussian and p_value < 0.001:
            score += 5

        # PRNU indicator (high = real camera, low = AI)
        if prnu_indicator > 0.1:
            score -= 10
        elif prnu_indicator < 0.01:
            score += 5

        # High kurtosis (heavy tails) is more typical of real images
        if avg_kurt > 5:
            score -= 8

        # --- Visualization ---
        noise_gray = np.mean(noise, axis=2)
        noise_norm = (noise_gray - noise_gray.min()) / (noise_gray.max() - noise_gray.min() + 1e-6)
        noise_norm = np.clip(noise_norm * 3, 0, 1)  # Amplify contrast

        vis_img = Image.fromarray((noise_norm * 255).astype(np.uint8))
        buffered = io.BytesIO()
        vis_img.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            'score': min(max(int(score), 0), 100),
            'details': {'metrics': metrics},
            'visualization': img_str
        }

    except Exception as e:
        return {
            'score': 50,
            'details': {'metrics': {}},
            'visualization': ''
        }
