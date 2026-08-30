import os
import time
import traceback
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

from analyzers.metadata_analyzer import analyze_metadata
from analyzers.wavelet_analyzer import analyze_wavelet
from analyzers.clip_analyzer import analyze_clip
from analyzers.neural_analyzer import analyze_neural
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

    O analisador Neural (Hugging Face pré-treinado) é o mais confiável e tem
    peso dominante. Os demais analisadores funcionam como evidências forenses
    complementares para refinar o score em casos limítrofes.
    """
    base_weights = {
        'metadata':    0.15,   # dinâmico — aumentado se há sinal EXIF forte
        'noise':       0.08,   # ruído de sensor Poisson-Gaussiano
        'spectral':    0.04,   # FFT — sinal fraco; peso mínimo
        'statistical': 0.12,   # entropia e Lei de Benford
        'wavelet':     0.10,   # DWT multi-escala
        'artifacts':   0.03,   # gradientes de borda e bokeh
        'clip':        0.12,   # espaço latente multimodal
        'neural':      0.60,   # detector pré-treinado HF — peso dominante
    }

    # --- Ajuste dinâmico do metadata ---
    meta = analyses.get('metadata', {})
    has_meta_signal = meta.get('has_signal', False)
    meta_score = meta.get('score', 50)

    if has_meta_signal and meta_score >= 90:
        base_weights['metadata'] = 0.40  # tag de IA forte → aumenta muito
    elif has_meta_signal and meta_score <= 15:
        base_weights['metadata'] = 0.30  # EXIF autentico de câmera → sinal forte
    elif not has_meta_signal:
        base_weights['metadata'] = 0.0   # sem EXIF → exclui do cálculo

    # --- Boost dinâmico do neural quando certeza é Alta ---
    neural = analyses.get('neural', {})
    if neural and not neural.get('failed'):
        n_metrics = neural.get('details', {}).get('metrics', {})
        certainty = n_metrics.get('certeza_classificacao', 'Baixa')
        if certainty == 'Alta':
            base_weights['neural'] = 0.72   # boost: certeza alta → ainda mais dominante
        elif certainty == 'Média':
            base_weights['neural'] = 0.60   # mantém peso normal

    total_score = 0.0
    total_weight = 0.0

    for key, weight in base_weights.items():
        # Analisadores que falharam são EXCLUÍDOS do composite.
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
    for key in ['noise', 'spectral', 'statistical', 'wavelet', 'artifacts', 'clip', 'neural']:
        if key in analyses and 'score' in analyses[key] and not analyses[key].get('failed'):
            scores.append(analyses[key]['score'])

    meta = analyses.get('metadata', {})
    if meta.get('has_signal', False) and 'score' in meta and not meta.get('failed'):
        scores.append(meta['score'])

    if len(scores) < 3:
        return 'low', 0.0

    classifications = []
    for s in scores:
        if s <= 38:
            classifications.append('real')
        elif s >= 62:
            classifications.append('ai')
        else:
            classifications.append('inconclusive')

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
        if p_corr > 0.036:
            key_findings.append("Ruído com correlação física Poisson-Gaussiana (típico de sensor fotográfico)")
        elif p_corr < -0.05:
            key_findings.append("Distribuição de ruído não-física (indicador de síntese algorítmica)")
        if b_noise >= 8.087:
            key_findings.append("Desvio-padrão do canal azul elevado — padrão consistente com síntese algorítmica")

    if 'spectral' in analyses:
        spec = analyses['spectral']
        metrics = spec.get('details', {}).get('metrics', {})
        flatness = metrics.get('hf_spectral_flatness', 0.5)
        peaks = metrics.get('anomalous_peaks', 0)
        alpha = metrics.get('spectral_slope_alpha', 1.0)
        r2 = metrics.get('power_law_fit_r2', 1.0)
        if flatness >= 0.9705:
            key_findings.append("Planura espectral de alta frequência elevada — forte indicador de síntese por IA (AUC=0.875)")
        elif 0.85 <= alpha <= 1.30 and r2 > 0.989:
            key_findings.append("Decaimento espectral de Fourier aderente à lei de potência óptica natural (1/f)")
        elif peaks >= 3:
            key_findings.append("Picos harmônicos periódicos detectados no espectro de Fourier")

    if 'statistical' in analyses:
        stat = analyses['statistical']
        metrics = stat.get('details', {}).get('metrics', {})
        r_entropy = metrics.get('r_entropy', 0)
        g_entropy = metrics.get('g_entropy', 0)
        avg_entropy = metrics.get('avg_entropy', 0)
        benford_corr = metrics.get('benford_correlation', 1.0)
        benford_dev = metrics.get('benford_deviation', 0)
        if r_entropy >= 7.104 and g_entropy >= 7.038:
            key_findings.append("Entropia dos canais RGB alta e equilibrada — padrão de imagem fotográfica natural (AUC=0.847)")
        elif r_entropy < 7.104 or g_entropy < 7.038:
            key_findings.append("Entropia de canal reduzida — frequente em imagens sintéticas (limiar calibrado)")
        if benford_dev >= 0.137:
            key_findings.append("Desvio da Lei de Benford acima do limiar calibrado — indicador de síntese algorítmica")
        elif benford_corr > 0.973:
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

    if 'neural' in analyses and not analyses['neural'].get('failed'):
        neural_data = analyses['neural']
        n_metrics = neural_data.get('details', {}).get('metrics', {})
        ai_prob = n_metrics.get('probabilidade_ai', 0.5)
        real_prob = n_metrics.get('probabilidade_real', 0.5)
        model_name = n_metrics.get('modelo_pretreinado', 'Vision Transformer (HF)')
        if ai_prob >= 0.70:
            key_findings.append(f"Detector Neural ({model_name}) identificou padrões de geração sintética com probabilidade de {ai_prob:.1%}")
        elif real_prob >= 0.70:
            key_findings.append(f"Detector Neural ({model_name}) classificou como foto autêntica com probabilidade de {real_prob:.1%}")

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
        'version': '2.2.0-transfer-learning',
        'analyzers': ['metadata', 'wavelet', 'spectral', 'noise', 'statistical', 'artifacts', 'clip', 'neural']
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
        ext = os.path.splitext(file.filename)[1].lower() if '.' in file.filename else '.png'
        temp_name = f"upload_{int(time.time() * 1000)}_{os.urandom(4).hex()}{ext}"
        original_path = os.path.join(UPLOAD_FOLDER, temp_name)

        file.save(original_path)
        file_size = os.path.getsize(original_path)

        with Image.open(original_path) as img:
            image_format = img.format or 'Unknown'
            image_size = img.size
            image_mode = img.mode

        analyses = {}
        analyzer_configs = [
            ('metadata', analyze_metadata, original_path),
            ('neural', analyze_neural, original_path),
            ('clip', analyze_clip, original_path),
            ('wavelet', analyze_wavelet, original_path),
            ('spectral', analyze_spectral, original_path),
            ('noise', analyze_noise, original_path),
            ('statistical', analyze_statistical, original_path),
            ('artifacts', analyze_artifacts, original_path),
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

        score = calculate_composite_score(analyses)
        verdict, verdict_level = get_verdict(score)

        confidence, agreement = calculate_concordance(analyses)

        summary = generate_summary(score, verdict, confidence, analyses)

        if failed_analyzers:
            summary['text'] += (
                f" Atenção: {len(failed_analyzers)} analisador(es) falharam durante o "
                f"processamento ({', '.join(failed_analyzers)}) e foram excluídos do "
                f"cálculo — o resultado reflete apenas os analisadores bem-sucedidos."
            )

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
        if original_path and os.path.exists(original_path):
            try:
                os.remove(original_path)
            except Exception:
                pass


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)