import numpy as np
from scipy import ndimage
from PIL import Image


def analyze_artifacts(image_path: str) -> dict:
    """
    Analyzes physical optical artifacts, depth of field gradients, edges, and latent patch boundaries.
    Features evaluated:
    - Optical depth of field & edge sharpness variance (physical lens focus vs synthetic sharpness)
    - Sobel edge orientation coherence & entropy
    - Authentic JPEG 8x8 DCT grid boundary detection (when applicable)
    - Texture variance across spatial blocks
    Returns: {'score': int, 'details': {'metrics': {...}}}
    """
    try:
        with Image.open(image_path) as src:
            source_format = (src.format or '').upper()
            img = src.convert('L')
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape

        # --- JPEG 8x8 Grid Boundary Ratio ---
        # Calculado ANTES do redimensionamento: o resize destrói o alinhamento da
        # grade 8x8 e fazia a métrica ficar ~1.00 em qualquer foto grande.
        # Usa o formato real do arquivo (não a extensão).
        # Mede compressão JPEG, não a origem da imagem — por isso é só informativo.
        is_jpeg = source_format == 'JPEG'
        jpeg_grid_ratio = 1.0
        if is_jpeg and h >= 32 and w >= 32:
            col_diffs = np.abs(np.diff(arr, axis=1)).mean(axis=0)  # diff entre col c-1 e c, c=1..w-1
            row_diffs = np.abs(np.diff(arr, axis=0)).mean(axis=1)
            col_idx = np.arange(1, w)
            row_idx = np.arange(1, h)
            boundary = np.concatenate([col_diffs[col_idx % 8 == 0], row_diffs[row_idx % 8 == 0]])
            non_boundary = np.concatenate([col_diffs[col_idx % 8 != 0], row_diffs[row_idx % 8 != 0]])
            avg_nb = float(np.mean(non_boundary)) if non_boundary.size else 0.0
            if boundary.size and avg_nb > 0:
                jpeg_grid_ratio = float(np.mean(boundary) / avg_nb)

        if max(h, w) > 1536:
            scale = 1536 / max(h, w)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            h, w = arr.shape

        metrics = {}

        # --- 1. Edge Detection & Sharpness Distribution (Optical Bokeh / DoF) ---
        dx = ndimage.sobel(arr, axis=1)
        dy = ndimage.sobel(arr, axis=0)
        mag = np.hypot(dx, dy)

        edge_threshold = np.percentile(mag, 85)
        strong_edges = mag > edge_threshold
        edge_magnitudes = mag[strong_edges]

        if len(edge_magnitudes) > 100:
            edge_cv = float(np.std(edge_magnitudes) / (np.mean(edge_magnitudes) + 1e-6))
            edge_skew = float(np.mean((edge_magnitudes - np.mean(edge_magnitudes)) ** 3) /
                              (np.std(edge_magnitudes) ** 3 + 1e-6))
        else:
            edge_cv = 0.55
            edge_skew = 0.5

        metrics['edge_sharpness_cv'] = round(edge_cv, 4)
        metrics['edge_sharpness_skew'] = round(edge_skew, 4)

        # --- 2. Edge Orientation Entropy ---
        orientations = np.arctan2(dy, dx)
        strong_orientations = orientations[strong_edges]

        if len(strong_orientations) > 100:
            hist, _ = np.histogram(strong_orientations, bins=36, range=(-np.pi, np.pi))
            hist_norm = hist / np.sum(hist)
            hist_pos = hist_norm[hist_norm > 0]
            edge_orientation_entropy = float(-np.sum(hist_pos * np.log2(hist_pos)))
            orientation_coherence = float(np.var(hist_norm))
        else:
            edge_orientation_entropy = 4.5
            orientation_coherence = 0.001

        metrics['edge_orientation_entropy'] = round(edge_orientation_entropy, 4)
        metrics['orientation_coherence'] = round(orientation_coherence, 6)

        # --- 3. Spatial Texture Variance (Depth & Multi-Layer Scene Complexity) ---
        block_h, block_w = max(h // 4, 1), max(w // 4, 1)
        block_contrasts = []
        for i in range(4):
            for j in range(4):
                y_start = i * block_h
                y_end = min((i + 1) * block_h, h)
                x_start = j * block_w
                x_end = min((j + 1) * block_w, w)
                block = arr[y_start:y_end, x_start:x_end]
                if block.size > 0:
                    block_contrasts.append(float(np.std(block)))

        contrast_var = float(np.var(block_contrasts)) if block_contrasts else 0.0
        contrast_cv = float(np.std(block_contrasts) / (np.mean(block_contrasts) + 1e-6)) if block_contrasts else 0.0
        metrics['texture_variance'] = round(contrast_var, 2)
        metrics['texture_cv'] = round(contrast_cv, 4)

        metrics['jpeg_grid_ratio'] = round(jpeg_grid_ratio, 4)
        metrics['is_jpeg_source'] = is_jpeg

        # --- 5. Scoring apenas por anomalias extremas ---
        # As regras antigas não separavam as classes no dataset de calibração
        # (AUC do score = 0.40): edge_cv, texture_cv e entropia de orientação
        # tiveram faixas praticamente iguais em fotos reais e imagens de IA, e o
        # bônus do grid JPEG media o formato do arquivo, não a origem.
        # Agora o score parte de 50 (neutro) e só sobe em casos extremos.
        # Este analisador é INFORMATIVO (peso 0 no composto) até ser recalibrado.
        score = 50.0

        if edge_cv < 0.20:
            score += 10  # Nitidez de bordas uniformemente constante
        if contrast_cv < 0.10:
            score += 8   # Cena sem nenhuma variação de contraste entre regiões
        if edge_orientation_entropy < 3.5:
            score += 8   # Orientações de borda fortemente enviesadas

        final_score = int(round(min(max(score, 0), 100)))

        return {
            'score': final_score,
            'informational': True,
            'details': {
                'metrics': metrics
            }
        }

    except Exception as e:
        return {
            'score': 50,
            # 'failed' sinaliza ao app.py que este resultado NÃO é válido e deve
            # ser excluído do score composto, da concordância e do resumo.
            'failed': True,
            'details': {'metrics': {}, 'error': str(e)}
        }
