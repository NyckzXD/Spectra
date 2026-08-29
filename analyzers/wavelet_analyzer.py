import io
import base64
import numpy as np
from scipy import ndimage, stats
from PIL import Image


def _compute_detail_scales(gray: np.ndarray):
    """
    Compute multi-scale detail residuals via Difference-of-Gaussians (Laplacian pyramid).

    Returns three detail arrays:
      d0 — fine detail   (original  − σ=1.5) : sensor noise, micro-texture
      d1 — medium detail (σ=1.5     − σ=4.0) : material/surface texture
      d2 — coarse detail (σ=4.0     − σ=10.0): large structural transitions
    """
    g0 = gray.astype(np.float64)
    g1 = ndimage.gaussian_filter(g0, sigma=1.5)
    g2 = ndimage.gaussian_filter(g0, sigma=4.0)
    g3 = ndimage.gaussian_filter(g0, sigma=10.0)
    return g0 - g1, g1 - g2, g2 - g3


def analyze_wavelet(image_path: str) -> dict:
    """
    Multi-scale wavelet/Laplacian residual analysis for AI image detection.
    Replaces ELA with a fundamentally better approach: instead of measuring
    JPEG recompression inconsistency, this analyzer measures the statistical
    properties of image detail at three frequency scales.

    Key forensic signals:
    - Kurtosis of fine-scale detail coefficients:
        Real cameras produce leptokurtic (heavy-tailed) detail due to sparse
        edges + photon shot noise. AI diffusion images produce near-Gaussian
        (lower kurtosis) smooth detail.
    - Cross-scale energy ratio (d0 / d1):
        Natural images follow a ~1/f² energy decay across scales.
        AI images often have abnormal scale-to-scale energy distribution.
    - Spatial coefficient of variation of fine detail energy:
        Real scenes have high spatial heterogeneity (focus planes, bokeh).
        AI images tend toward uniform detail energy distributions.
    - Inter-channel correlation in fine detail:
        Real cameras (CFA demosaicing) produce partially correlated R/G/B
        detail residuals. AI synthesis patterns differ.
    - Tail ratio of fine detail (P95 / P50):
        Measures heavy-tailedness robustly; higher = more edge/noise outliers.

    Returns: {'score': int, 'details': {'metrics': {...}}, 'visualization': 'base64string'}
    """
    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img, dtype=np.float32)
        h, w, _ = arr.shape

        if max(h, w) > 1536:
            scale = 1536 / max(h, w)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            h, w, _ = arr.shape

        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

        d0, d1, d2 = _compute_detail_scales(gray)

        metrics = {}

        # --- 1. Kurtosis of detail coefficients at each scale ---
        # Real camera images: fine-scale (d0) has HIGH kurtosis (4–15+) due to:
        #   a) sparse edge structure → heavy-tailed coefficient distribution
        #   b) Poisson/Gaussian photon noise → slight super-Gaussianity
        # AI diffusion images: smoother, near-Gaussian detail → kurtosis 0–3
        d0_flat = d0.ravel()
        d1_flat = d1.ravel()
        d2_flat = d2.ravel()

        kurt_d0 = float(stats.kurtosis(d0_flat))
        kurt_d1 = float(stats.kurtosis(d1_flat))
        kurt_d2 = float(stats.kurtosis(d2_flat))

        metrics['kurtosis_fine'] = round(kurt_d0, 4)
        metrics['kurtosis_medium'] = round(kurt_d1, 4)
        metrics['kurtosis_coarse'] = round(kurt_d2, 4)

        # --- 2. Cross-scale energy ratio ---
        # Natural images: energy decays ~4–8× per octave (related to 1/f² spatial spectrum).
        # AI images: can show abnormal energy concentration or flattening across scales.
        energy_d0 = float(np.mean(d0 ** 2))
        energy_d1 = float(np.mean(d1 ** 2))
        energy_d2 = float(np.mean(d2 ** 2))

        # Fine-to-medium ratio (real photos: typically 3–18×)
        e01_ratio = float(energy_d0 / (energy_d1 + 1e-8))
        # Medium-to-coarse ratio
        e12_ratio = float(energy_d1 / (energy_d2 + 1e-8))

        metrics['energy_fine'] = round(energy_d0, 6)
        metrics['energy_medium'] = round(energy_d1, 6)
        metrics['energy_coarse'] = round(energy_d2, 6)
        metrics['energy_ratio_fine_medium'] = round(e01_ratio, 4)
        metrics['energy_ratio_medium_coarse'] = round(e12_ratio, 4)

        # --- 3. Spatial block variance of fine detail energy ---
        # Natural scenes: high spatial variance (sharp foreground vs blurry background,
        # multi-object depth, DoF transitions).
        # AI images: more spatially homogeneous fine detail energy distribution.
        block_size = 32
        block_energies = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = d0[y:y + block_size, x:x + block_size]
                block_energies.append(float(np.mean(block ** 2)))

        if len(block_energies) > 4:
            spatial_cv = float(np.std(block_energies) / (np.mean(block_energies) + 1e-8))
        else:
            spatial_cv = 0.5

        metrics['spatial_detail_cv'] = round(spatial_cv, 4)

        # --- 4. Inter-channel correlation of fine detail ---
        # Real cameras: R/G/B detail channels partially correlated (CFA demosaicing residuals,
        # chromatic aberration, shared edge structures).
        # AI synthesis: channel detail patterns can diverge from this physical model.
        sample_size = min(40000, h * w)
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(h * w, sample_size, replace=False)

        d0_r = (arr[:, :, 0].astype(np.float64)
                - ndimage.gaussian_filter(arr[:, :, 0].astype(np.float64), 1.5)).ravel()[idx]
        d0_g = (arr[:, :, 1].astype(np.float64)
                - ndimage.gaussian_filter(arr[:, :, 1].astype(np.float64), 1.5)).ravel()[idx]
        d0_b = (arr[:, :, 2].astype(np.float64)
                - ndimage.gaussian_filter(arr[:, :, 2].astype(np.float64), 1.5)).ravel()[idx]

        corr_rg = float(np.corrcoef(d0_r, d0_g)[0, 1])
        corr_rb = float(np.corrcoef(d0_r, d0_b)[0, 1])
        corr_gb = float(np.corrcoef(d0_g, d0_b)[0, 1])
        avg_channel_corr = float((corr_rg + corr_rb + corr_gb) / 3.0)

        metrics['detail_corr_rg'] = round(corr_rg, 4)
        metrics['detail_corr_rb'] = round(corr_rb, 4)
        metrics['detail_corr_gb'] = round(corr_gb, 4)
        metrics['avg_detail_channel_corr'] = round(avg_channel_corr, 4)

        # --- 5. Tail-weight of fine detail (P95 / P50 ratio) ---
        # Measures heavy-tailedness more robustly than kurtosis for large, non-stationary arrays.
        # Real photos: higher tail ratio (extreme outliers from sharp edges + noise peaks)
        # AI images: lower tail ratio (smoother, more centrally concentrated detail)
        abs_d0 = np.abs(d0_flat)
        p95 = float(np.percentile(abs_d0, 95))
        p50 = float(np.percentile(abs_d0, 50))
        tail_ratio = float(p95 / (p50 + 1e-8))

        metrics['fine_detail_tail_ratio'] = round(tail_ratio, 4)

        # --- 6. Symmetrical Calibrated Scoring Model ---
        score = 50.0

        # Kurtosis of fine detail: HIGH → real sensor noise + sparse edges
        if kurt_d0 > 8.0:
            score -= 20  # Strongly leptokurtic — real sensor / optical edge signature
        elif kurt_d0 > 4.0:
            score -= 10  # Moderately leptokurtic
        elif kurt_d0 < 1.5:
            score += 18  # Near-Gaussian detail — synthetic VAE/UNet smoothing
        elif kurt_d0 < 2.5:
            score += 8

        # Cross-scale energy ratio: natural 1/f² decay range
        if 3.5 <= e01_ratio <= 18.0:
            score -= 10  # Natural energy pyramid decay
        elif e01_ratio < 1.8:
            score += 12  # Abnormally flat (diffusion noise floor artifact)
        elif e01_ratio > 40.0:
            score += 6   # Over-sharpened (AI post-processing artifact)

        # Spatial heterogeneity: natural depth/focus variation
        if spatial_cv > 0.90:
            score -= 12  # High variance → natural multi-plane scene
        elif spatial_cv > 0.55:
            score -= 5
        elif spatial_cv < 0.25:
            score += 14  # Homogeneous → synthetic flat composition
        elif spatial_cv < 0.40:
            score += 6

        # Tail ratio: heavy-tailed = real edges + sensor noise outliers
        if tail_ratio > 18.0:
            score -= 8   # Heavy-tailed outliers (natural edge structure)
        elif tail_ratio < 7.0:
            score += 10  # Light tails (AI-smoothed fine detail)

        final_score = int(round(min(max(score, 0), 100)))

        # --- Visualization: fine detail (d0) heatmap ---
        d0_abs = np.abs(d0)
        p99 = np.percentile(d0_abs, 99)
        vis_norm = np.clip(d0_abs / (p99 + 1e-8), 0.0, 1.0)

        # Blue → Cyan → Yellow colormap (highlights fine detail structure)
        r = np.clip(vis_norm * 2.0 - 0.3, 0, 1)
        g = np.clip(vis_norm * 1.8, 0, 1)
        b = np.clip(1.0 - vis_norm * 1.8, 0, 1)

        color_vis = np.stack([
            (r * 255).astype(np.uint8),
            (g * 255).astype(np.uint8),
            (b * 255).astype(np.uint8)
        ], axis=-1)

        vis_img = Image.fromarray(color_vis)
        buffered = io.BytesIO()
        vis_img.save(buffered, format='PNG', optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {
            'score': final_score,
            'details': {
                'metrics': metrics
            },
            'visualization': img_str
        }

    except Exception as e:
        return {
            'score': 50,
            'details': {'metrics': {}, 'error': str(e)},
            'visualization': ''
        }
