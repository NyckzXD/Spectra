import os
import io
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
        'spectral': 0.25,
        'noise': 0.20,
        'statistical': 0.15,
        'ela': 0.10,
        'artifacts': 0.05
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


def get_verdict(score):
    """Get verdict text and level based on score."""
    if score <= 25:
        return 'Provavelmente Real', 'real'
    elif score <= 50:
        return 'Inconclusivo (tendência real)', 'inconclusive'
    elif score <= 75:
        return 'Suspeito (tendência AI)', 'suspect'
    else:
        return 'Provavelmente Gerado por AI', 'ai'


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint. Accepts image upload and runs all analyzers."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Nenhuma imagem enviada'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Nenhum arquivo selecionado'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': f'Formato não suportado. Use: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save temporarily for analysis
        temp_path = os.path.join(UPLOAD_FOLDER, 'temp_analysis.png')
        
        # Open with PIL to validate and get info
        image = Image.open(file.stream)
        image_format = image.format or 'Unknown'
        image_size = image.size
        
        # Save as PNG for consistent analysis
        image.save(temp_path, 'PNG')
        
        # Also save original for metadata analysis (preserve original format)
        file.stream.seek(0)
        original_path = os.path.join(UPLOAD_FOLDER, f'temp_original{os.path.splitext(file.filename)[1]}')
        file.save(original_path)
        
        # Get file size
        file_size = os.path.getsize(original_path)
        
        # Run all analyzers
        analyses = {}
        
        try:
            analyses['metadata'] = analyze_metadata(original_path)
        except Exception as e:
            analyses['metadata'] = {'score': 50, 'details': {}, 'findings': [f'Erro na análise: {str(e)}']}
            traceback.print_exc()
        
        try:
            analyses['ela'] = analyze_ela(temp_path)
        except Exception as e:
            analyses['ela'] = {'score': 50, 'details': {}, 'error': str(e)}
            traceback.print_exc()
        
        try:
            analyses['spectral'] = analyze_spectral(temp_path)
        except Exception as e:
            analyses['spectral'] = {'score': 50, 'details': {}, 'error': str(e)}
            traceback.print_exc()
        
        try:
            analyses['noise'] = analyze_noise(temp_path)
        except Exception as e:
            analyses['noise'] = {'score': 50, 'details': {}, 'error': str(e)}
            traceback.print_exc()
        
        try:
            analyses['statistical'] = analyze_statistical(temp_path)
        except Exception as e:
            analyses['statistical'] = {'score': 50, 'details': {}, 'error': str(e)}
            traceback.print_exc()
        
        try:
            analyses['artifacts'] = analyze_artifacts(temp_path)
        except Exception as e:
            analyses['artifacts'] = {'score': 50, 'details': {}, 'error': str(e)}
            traceback.print_exc()
        
        # Calculate composite score
        score = calculate_composite_score(analyses)
        verdict, verdict_level = get_verdict(score)
        
        # Clean up temp files
        try:
            os.remove(temp_path)
            os.remove(original_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'score': score,
            'verdict': verdict,
            'verdict_level': verdict_level,
            'analyses': analyses,
            'image_info': {
                'filename': file.filename,
                'format': image_format,
                'size': list(image_size),
                'file_size': file_size
            }
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Erro ao processar imagem: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
