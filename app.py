import os
import time
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

from analyzers.metadata_analyzer import analyze_metadata
from analyzers.wavelet_analyzer import analyze_wavelet
from analyzers.clip_analyzer import analyze_clip
from analyzers.spectral_analyzer import analyze_spectral
from analyzers.noise_analyzer import analyze_noise
from analyzers.statistical_analyzer import analyze_statistical
from analyzers.artifact_analyzer import analyze_artifacts

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_composite_score(analyses):
    """
    Calculate dynamically weighted composite score from all forensic analyses.
    Adapts weights based on metadata signal presence to prevent biasing images without EXIF.

    Pesos calibrados empiricamente em 28/08/2026 a partir de dataset rotulado
    (9 imagens reais + 8 imagens de IA). AUC por analisador no dataset:
      statistical  AUC=0.764 (mais discriminativo) -> peso aumentado
      noise        AUC=0.563 (sinal moderado)       -> peso mantido
      ela          AUC=0.514 (sinal fraco)           -> peso mantido
      spectral     AUC=0.403 (parcialmente invertido)-> peso reduzido
      artifacts    AUC=0.403 (parcialmente invertido)-> peso reduzido
      metadata     AUC=0.500 (ruido no dataset*)     -> mantém lógica dinâmica
    * Todas as imagens do dataset tinham score=50 por falta de EXIF;
      o metadata não foi testado com imagens que têm EXIF real da câmera.
    """
    base_weights = {
        'metadata': 0.22,    # dinâmico — mantém lógica original (ver abaixo)
        'noise': 0.16,       # AUC 0.563 no dataset calibrado
        'spectral': 0.07,    # AUC 0.403 — parcialmente invertido; peso reduzido
        'statistical': 0.27, # AUC 0.764 — analisador mais discriminativo no dataset
        'wavelet': 0.23,     # substitui ELA — DWT multi-escala, mais discriminativo (peso aumentado)
        'artifacts': 0.05,   # AUC 0.403 — parcialmente invertido; peso reduzido
        'clip': 0.35,        # CLIP ViT-B/32 — maior peso (mais preciso); excluído se open_clip indisponível
    }

    meta = analyses.get('metadata', {})
    has_meta_signal = meta.get('has_signal', False)
    meta_score = meta.get('score', 50)

    # Dynamic adjustment based on metadata signal
    if has_meta_signal and meta_score >= 90:
        # Decisive AI signatures (prompts, generator tags, C2PA)
        base_weights['metadata'] = 0.45
    elif has_meta_signal and meta_score <= 15:
        # Decisive authentic camera EXIF (Make, Model, ISO, Exposure, GPS)
        base_weights['metadata'] = 0.35
    elif not has_meta_signal:
        # Neutral metadata (stripped EXIF from web/social media)
        # Exclude metadata from pulling the composite score
        base_weights['metadata'] = 0.0

    total_score = 0.0
    total_weight = 0.0

    for key, weight in base_weights.items():
        # Analisadores que falharam (exceção interna) são EXCLUÍDOS do composite:
        # antes, um score neutro (50) de fallback entrava como se fosse uma medição
        # válida, distorcendo o resultado final sem avisar o usuário.
        if key in analyses and 'score' in analyses[key] and not analyses[key].get('failed'):
            total_score += weight * analyses[key]['score']
            total_weight += weight

    if total_weight > 0:
        composite = total_score / total_weight
        return int(round(min(max(composite, 0), 100)))
    return 50  # fallback inconclusive


def calculate_concordance(analyses):
    """
    Calculate inter-analyzer concordance and agreement ratio.
    Returns confidence level ('high', 'medium', 'low') and agreement ratio.
    """
    scores = []
    # Collect visual and active forensic analyzers (exclui analisadores que falharam)
    for key in ['noise', 'spectral', 'statistical', 'wavelet', 'artifacts', 'clip']:
        if key in analyses and 'score' in analyses[key] and not analyses[key].get('failed'):
            scores.append(analyses[key]['score'])

    meta = analyses.get('metadata', {})
    if meta.get('has_signal', False) and 'score' in meta and not meta.get('failed'):
        scores.append(meta['score'])

    if len(scores) < 3:
        return 'low', 0.0

    # Classify each score
    classifications = []
    for s in scores:
        if s <= 38:
            classifications.append('real')
        elif s >= 62:
            classifications.append('ai')
        else:
            classifications.append('inconclusive')

    # Count agreement
    from collections import Counter
    counts = Counter(classifications)
    most_common, most_count = counts.most_common(1)[0]

    agreement_ratio = most_count / len(scores)
    score_std = float(np.std(scores)) if len(scores) > 1 else 30.0

    if agreement_ratio >= 0.75 and score_std < 18:
        return 'high', agreement_ratio
    elif agreement_ratio >= 0.55 or score_std < 24:
        return 'medium', agreement_ratio
    else:
        return 'low', agreement_ratio


def get_verdict(score):
    """Get human-readable verdict and category level based on calibrated score."""
    if score <= 25:
        return 'Provavelmente Autêntica (Foto Real)', 'real'
    elif score <= 40:
        return 'Tendência Autêntica (Baixo Risco IA)', 'inconclusive-real'
    elif score <= 59:
        return 'Inconclusivo (Sinais Mistos)', 'inconclusive'
    elif score <= 74:
        return 'Suspeito — Tendência IA', 'suspect'
    else:
        return 'Provavelmente Gerada por IA', 'ai'


def generate_summary(score, verdict, confidence, analyses):
    """Generate an informative technical summary of the forensic findings."""
    summary_parts = []

    if score <= 25:
        summary_parts.append(
            "A análise forense multi-espectral indica alta probabilidade de a imagem ser uma captura autêntica de câmera física."
        )
    elif score <= 40:
        summary_parts.append(
            "Os indicadores forenses apontam predominância de características naturais de captura óptica, com baixo indício de síntese por IA."
        )
    elif score <= 59:
        summary_parts.append(
            "Os padrões forenses apresentam sinais mistos ou insuficientes para uma classificação categórica entre foto real e geração por IA."
        )
    elif score <= 74:
        summary_parts.append(
            "Foram detectadas anomalias estatísticas e padrões de síntese característicos de modelos de difusão de inteligência artificial."
        )
    else:
        summary_parts.append(
            "Múltiplos analisadores identificaram fortes assinaturas forenses de síntese algorítmica por IA (Difusão/VAE/Redes Neurais)."
        )

    # Key findings
    key_findings = []

    if 'metadata' in analyses:
        meta = analyses['metadata']
        if meta.get('findings'):
            for f in meta['findings'][:2]:
                key_findings.append(f)

    if 'noise' in analyses:
        noise = analyses['noise']
        metrics = noise.get('details', {}).get('metrics', {})
        p_corr = metrics.get('poisson_correlation', 0)
        b_noise = metrics.get('b_noise_std', 0)
        # Limiar poisson_correlation calibrado: AUC 0.597, threshold Youden=0.036
        if p_corr > 0.036:
            key_findings.append("Ruído com correlação física Poisson-Gaussiana (típico de sensor fotográfico)")
        elif p_corr < -0.05:
            key_findings.append("Distribuição de ruído não-física (indicador de síntese algorítmica)")
        # b_noise_std: AUC 0.750, threshold Youden=8.087 — valor alto → IA
        if b_noise >= 8.087:
            key_findings.append("Desvio-padrão do canal azul elevado — padrão consistente com síntese algorítmica")

    if 'spectral' in analyses:
        spec = analyses['spectral']
        metrics = spec.get('details', {}).get('metrics', {})
        # hf_spectral_flatness: métrica mais discriminativa do dataset (AUC=0.875)
        # Threshold Youden calibrado: 0.9705 (higher_is_ai)
        flatness = metrics.get('hf_spectral_flatness', 0.5)
        peaks = metrics.get('anomalous_peaks', 0)
        alpha = metrics.get('spectral_slope_alpha', 1.0)
        r2 = metrics.get('power_law_fit_r2', 1.0)
        if flatness >= 0.9705:
            key_findings.append("Planura espectral de alta frequência elevada — forte indicador de síntese por IA (AUC=0.875)")
        elif 0.85 <= alpha <= 1.30 and r2 > 0.989:
            # r2 threshold calibrado: Youden=0.989 (higher_is_real, AUC=0.625)
            key_findings.append("Decaimento espectral de Fourier aderente à lei de potência óptica natural (1/f)")
        elif peaks >= 3:
            key_findings.append("Picos harmônicos periódicos detectados no espectro de Fourier")

    if 'statistical' in analyses:
        stat = analyses['statistical']
        metrics = stat.get('details', {}).get('metrics', {})
        # r_entropy: AUC=0.847, threshold Youden=7.104 (higher_is_real)
        r_entropy = metrics.get('r_entropy', 0)
        g_entropy = metrics.get('g_entropy', 0)
        avg_entropy = metrics.get('avg_entropy', 0)
        benford_corr = metrics.get('benford_correlation', 1.0)
        benford_dev = metrics.get('benford_deviation', 0)
        if r_entropy >= 7.104 and g_entropy >= 7.038:
            key_findings.append("Entropia dos canais RGB alta e equilibrada — padrão de imagem fotográfica natural (AUC=0.847)")
        elif r_entropy < 7.104 or g_entropy < 7.038:
            key_findings.append("Entropia de canal reduzida — frequente em imagens sintéticas (limiar calibrado)")
        # benford_deviation: AUC=0.736, threshold=0.137 (higher_is_ai)
        if benford_dev >= 0.137:
            key_findings.append("Desvio da Lei de Benford acima do limiar calibrado — indicador de síntese algorítmica")
        elif benford_corr > 0.973:
            # benford_correlation: AUC=0.667, threshold Youden=0.973 (higher_is_real)
            key_findings.append("Gradientes aderem à Lei de Benford natural de superfícies físicas")

    if 'wavelet' in analyses:
        wav = analyses['wavelet']
        metrics = wav.get('details', {}).get('metrics', {})
        kurt_fine = metrics.get('kurtosis_fine', 3.0)
        spatial_cv = metrics.get('spatial_detail_cv', 0.5)
        tail_ratio = metrics.get('fine_detail_tail_ratio', 10.0)
        if kurt_fine > 8.0:
            key_findings.append("Kurtose elevada nos coeficientes de detalhe fino — assinatura de ruído de sensor fotográfico")
        elif kurt_fine < 1.5:
            key_findings.append("Coeficientes de detalhe fino quasi-Gaussianos — padrão de suavização sintética (VAE/UNet)")
        if spatial_cv < 0.25:
            key_findings.append("Distribuição espacial de detalhe homogênea — composição sintética detectada")
        elif spatial_cv > 0.90:
            key_findings.append("Alta variância espacial de detalhe — variação natural de foco e profundidade de campo")

    if 'clip' in analyses and not analyses['clip'].get('failed'):
        clip_data = analyses['clip']
        clip_metrics = clip_data.get('details', {}).get('metrics', {})
        diff = clip_metrics.get('similarity_diff', 0.0)
        mode = clip_data.get('details', {}).get('mode', 'unknown')
        mode_label = "(protótipos do dataset)" if mode == 'prototype' else "(texto — instale protótipos)"
        if diff > 0.04:
            key_findings.append(f"Embedding CLIP posicionado na região de imagens sintéticas no espaço latente {mode_label}")
        elif diff < -0.04:
            key_findings.append(f"Embedding CLIP posicionado na região de fotografias reais no espaço latente {mode_label}")

    if 'artifacts' in analyses:
        art = analyses['artifacts']
        metrics = art.get('details', {}).get('metrics', {})
        if metrics.get('edge_sharpness_cv', 0.5) > 0.62:
            key_findings.append("Gradiente de profundidade de campo óptico (bokeh/foco natural)")

    if confidence == 'high':
        summary_parts.append("Os analisadores forenses estão em forte concordância técnica.")
    elif confidence == 'low':
        summary_parts.append("Houve dispersão entre os analisadores — avalie os detalhes métricos individuais.")

    return {
        'text': ' '.join(summary_parts),
        'key_findings': key_findings[:5]
    }


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'version': '2.1.0-wavelet-clip',
        'analyzers': ['metadata', 'wavelet', 'spectral', 'noise', 'statistical', 'artifacts', 'clip']
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint. Accepts image upload and runs all analyzers."""
    start_time = time.time()

    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Nenhuma imagem enviada'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nenhum arquivo selecionado'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Formato não suportado. Use: {", ".join(sorted(ALLOWED_EXTENSIONS))}'
        }), 400

    original_path = None
    try:
        # Determine extension and unique temp filename
        ext = os.path.splitext(file.filename)[1].lower() if '.' in file.filename else '.png'
        temp_name = f"upload_{int(time.time() * 1000)}_{os.urandom(4).hex()}{ext}"
        original_path = os.path.join(UPLOAD_FOLDER, temp_name)

        # Save original intact file (preserves compression, EXIF, and native structure)
        file.save(original_path)
        file_size = os.path.getsize(original_path)

        # Validate with PIL and extract fundamental format info
        with Image.open(original_path) as img:
            image_format = img.format or 'Unknown'
            image_size = img.size
            image_mode = img.mode

        # Run all analyzers on the authentic source file
        analyses = {}
        analyzer_configs = [
            ('metadata', analyze_metadata, original_path),
            ('wavelet', analyze_wavelet, original_path),
            ('spectral', analyze_spectral, original_path),
            ('noise', analyze_noise, original_path),
            ('statistical', analyze_statistical, original_path),
            ('artifacts', analyze_artifacts, original_path),
            ('clip', analyze_clip, original_path),
        ]

        failed_analyzers = []
        for key, func, path in analyzer_configs:
            try:
                analyses[key] = func(path)
            except Exception as e:
                # 'failed': True marca explicitamente que este NÃO é um resultado válido.
                # calculate_composite_score e calculate_concordance excluem esses
                # analisadores do cálculo em vez de tratar o score 50 como neutro real.
                analyses[key] = {
                    'score': 50,
                    'failed': True,
                    'details': {'metrics': {}},
                    'findings': [f'Erro na análise: {str(e)}']
                }
                failed_analyzers.append(key)
                traceback.print_exc()

        # Calculate composite score with dynamic weighting
        score = calculate_composite_score(analyses)
        verdict, verdict_level = get_verdict(score)

        # Calculate concordance
        confidence, agreement = calculate_concordance(analyses)

        # Generate summary
        summary = generate_summary(score, verdict, confidence, analyses)

        if failed_analyzers:
            summary['text'] += (
                f" Atenção: {len(failed_analyzers)} analisador(es) falharam durante o "
                f"processamento ({', '.join(failed_analyzers)}) e foram excluídos do "
                f"cálculo — o resultado reflete apenas os analisadores bem-sucedidos."
            )

        # Processing time
        elapsed = round(time.time() - start_time, 2)

        return jsonify({
            'success': True,
            'score': score,
            'verdict': verdict,
            'verdict_level': verdict_level,
            'confidence': confidence,
            'agreement': round(agreement, 2),
            'summary': summary,
            'analyses': analyses,
            'failed_analyzers': failed_analyzers,
            'image_info': {
                'filename': file.filename,
                'format': image_format,
                'mode': image_mode,
                'size': list(image_size),
                'file_size': file_size
            },
            'processing_time': elapsed
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Erro ao processar imagem: {str(e)}'}), 500

    finally:
        # Clean up temporary uploaded file safely
        if original_path and os.path.exists(original_path):
            try:
                os.remove(original_path)
            except Exception:
                pass


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)