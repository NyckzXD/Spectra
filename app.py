import os
import time
import traceback
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from analyzers.metadata_analyzer import analyze_metadata
from analyzers.ela_analyzer import analyze_ela
from analyzers.spectral_analyzer import analyze_spectral
from analyzers.noise_analyzer import analyze_noise
from analyzers.statistical_analyzer import analyze_statistical
from analyzers.artifact_analyzer import analyze_artifacts

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_composite_score(analyses):
    """Calculate weighted composite score from all analyses."""
    weights = {
        'metadata': 0.25,
        'spectral': 0.20,
        'noise': 0.20,
        'statistical': 0.15,
        'ela': 0.12,
        'artifacts': 0.08
    }

    total_score = 0
    total_weight = 0

    for key, weight in weights.items():
        if key in analyses and 'score' in analyses[key]:
            total_score += weight * analyses[key]['score']
            total_weight += weight

    if total_weight > 0:
        return round(total_score / total_weight)
    return 50  # inconclusive


def calculate_concordance(analyses):
    """
    Calculate inter-analyzer concordance.
    Returns confidence level based on how much analyzers agree.
    """
    scores = []
    for key in ['metadata', 'ela', 'spectral', 'noise', 'statistical', 'artifacts']:
        if key in analyses and 'score' in analyses[key]:
            scores.append(analyses[key]['score'])

    if len(scores) < 3:
        return 'low', 0.0

    # Classify each score
    classifications = []
    for s in scores:
        if s <= 35:
            classifications.append('real')
        elif s >= 65:
            classifications.append('ai')
        else:
            classifications.append('inconclusive')

    # Count agreement
    from collections import Counter
    counts = Counter(classifications)
    most_common, most_count = counts.most_common(1)[0]

    agreement_ratio = most_count / len(scores)

    # Standard deviation of scores (lower = more agreement)
    score_std = float(np.std(scores)) if len(scores) > 1 else 50.0

    if agreement_ratio >= 0.8 and score_std < 20:
        return 'high', agreement_ratio
    elif agreement_ratio >= 0.6 or score_std < 25:
        return 'medium', agreement_ratio
    else:
        return 'low', agreement_ratio


def get_verdict(score):
    """Get verdict text and level based on score."""
    if score <= 25:
        return 'Provavelmente Autêntica', 'real'
    elif score <= 45:
        return 'Inconclusivo — tendência real', 'inconclusive-real'
    elif score <= 55:
        return 'Inconclusivo', 'inconclusive'
    elif score <= 75:
        return 'Suspeito — tendência AI', 'suspect'
    else:
        return 'Provavelmente Gerada por AI', 'ai'


def generate_summary(score, verdict, confidence, analyses):
    """Generate a human-readable summary of the analysis."""
    summary_parts = []

    if score <= 25:
        summary_parts.append(
            "A análise forense indica alta probabilidade de esta ser uma imagem autêntica (não gerada por IA)."
        )
    elif score <= 45:
        summary_parts.append(
            "A análise forense sugere que esta imagem provavelmente é autêntica, mas alguns indicadores são inconclusivos."
        )
    elif score <= 55:
        summary_parts.append(
            "A análise forense não conseguiu determinar com confiança se esta imagem é real ou gerada por IA."
        )
    elif score <= 75:
        summary_parts.append(
            "A análise forense detectou indicadores suspeitos compatíveis com geração por IA."
        )
    else:
        summary_parts.append(
            "A análise forense indica alta probabilidade de esta imagem ter sido gerada por inteligência artificial."
        )

    # Key findings
    key_findings = []

    if 'metadata' in analyses:
        meta = analyses['metadata']
        if meta.get('findings'):
            for f in meta['findings'][:2]:
                key_findings.append(f)

    if 'spectral' in analyses:
        spec = analyses['spectral']
        if spec.get('details', {}).get('metrics', {}).get('anomalous_peaks', 0) > 3:
            key_findings.append("Picos espectrais anômalos detectados (possível fingerprint de GAN)")

    if 'noise' in analyses:
        noise = analyses['noise']
        prnu = noise.get('details', {}).get('metrics', {}).get('prnu_indicator', 0)
        if prnu > 0.1:
            key_findings.append("Padrão PRNU de sensor detectado (indicador de câmera real)")

    if 'artifacts' in analyses:
        art = analyses['artifacts']
        cb = art.get('details', {}).get('metrics', {}).get('checkerboard_score', 0)
        if cb > 0.1:
            key_findings.append("Artefatos checkerboard detectados (fingerprint de GAN)")

    if confidence == 'high':
        summary_parts.append("Os analisadores estão em forte concordância.")
    elif confidence == 'low':
        summary_parts.append("Os analisadores divergem significativamente — resultado deve ser interpretado com cautela.")

    return {
        'text': ' '.join(summary_parts),
        'key_findings': key_findings[:5]
    }


import numpy as np  # needed for concordance calculation


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'version': '1.0.0-mvp',
        'analyzers': ['metadata', 'ela', 'spectral', 'noise', 'statistical', 'artifacts']
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

    try:
        # Save temporarily for analysis
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp_analysis.png')

        # Open with PIL to validate and get info
        image = Image.open(file.stream)
        image_format = image.format or 'Unknown'
        image_size = image.size
        image_mode = image.mode

        # Save as PNG for consistent analysis
        image.save(temp_path, 'PNG')

        # Also save original for metadata analysis (preserve original format)
        file.stream.seek(0)
        ext = os.path.splitext(file.filename)[1] if '.' in file.filename else '.png'
        original_path = os.path.join(UPLOAD_FOLDER, f'temp_original{ext}')
        file.save(original_path)

        # Get file size
        file_size = os.path.getsize(original_path)

        # Run all analyzers
        analyses = {}

        analyzer_configs = [
            ('metadata', analyze_metadata, original_path),
            ('ela', analyze_ela, temp_path),
            ('spectral', analyze_spectral, temp_path),
            ('noise', analyze_noise, temp_path),
            ('statistical', analyze_statistical, temp_path),
            ('artifacts', analyze_artifacts, temp_path),
        ]

        for key, func, path in analyzer_configs:
            try:
                analyses[key] = func(path)
            except Exception as e:
                analyses[key] = {'score': 50, 'details': {}, 'findings': [f'Erro na análise: {str(e)}']}
                traceback.print_exc()

        # Calculate composite score
        score = calculate_composite_score(analyses)
        verdict, verdict_level = get_verdict(score)

        # Calculate concordance
        confidence, agreement = calculate_concordance(analyses)

        # Generate summary
        summary = generate_summary(score, verdict, confidence, analyses)

        # Processing time
        elapsed = round(time.time() - start_time, 2)

        # Clean up temp files
        for p in [temp_path, original_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        return jsonify({
            'success': True,
            'score': score,
            'verdict': verdict,
            'verdict_level': verdict_level,
            'confidence': confidence,
            'agreement': round(agreement, 2),
            'summary': summary,
            'analyses': analyses,
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
