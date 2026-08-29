import os
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level singleton — O modelo pré-treinado do Hugging Face é carregado
# uma única vez na memória e reutilizado para todas as análises subsequentes.
# ---------------------------------------------------------------------------
_hf_pipeline = None
_hf_processor = None
_hf_model = None
_hf_device = None
_hf_loaded = False
_hf_load_error = None

# Modelo padrão especializado em detecção de imagens sintéticas (treinado em centenas de milhares de imagens)
DEFAULT_HF_MODEL = "umm-maybe/AI-image-detector"
FALLBACK_HF_MODEL = "dima806/ai_vs_real_image_detection"


def _try_load_hf_model():
    """
    Tenta carregar o modelo de detecção de IA pré-treinado do Hugging Face.
    Baixa e cacheia automaticamente no primeiro uso (~300MB).
    """
    global _hf_pipeline, _hf_processor, _hf_model, _hf_device, _hf_loaded, _hf_load_error

    if _hf_loaded:
        return _hf_model is not None or _hf_pipeline is not None

    _hf_loaded = True

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        _hf_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            model_id = DEFAULT_HF_MODEL
            _hf_processor = AutoImageProcessor.from_pretrained(model_id)
            _hf_model = AutoModelForImageClassification.from_pretrained(model_id)
            _hf_model.to(_hf_device)
            _hf_model.eval()
            return True
        except Exception as e1:
            # Tenta modelo alternativo se o primeiro falhar
            try:
                model_id = FALLBACK_HF_MODEL
                _hf_processor = AutoImageProcessor.from_pretrained(model_id)
                _hf_model = AutoModelForImageClassification.from_pretrained(model_id)
                _hf_model.to(_hf_device)
                _hf_model.eval()
                return True
            except Exception as e2:
                _hf_load_error = f"Falha ao carregar modelos Hugging Face ({e1} | {e2})"
                return False

    except ImportError:
        _hf_load_error = (
            "A biblioteca 'transformers' não está instalada. "
            "Execute: pip install transformers"
        )
        return False
    except Exception as e:
        _hf_load_error = str(e)
        return False


def analyze_neural(image_path: str) -> dict:
    """
    Análise de Visão Computacional Profunda via Modelo Especialista Pré-Treinado (Hugging Face).

    Utiliza um Vision Transformer (ViT/DeiT) treinado em centenas de milhares de imagens
    reais e sintéticas (Midjourney, Stable Diffusion, DALL-E, Flux, etc.).

    NÃO necessita de treinamento local nem de dataset prévio — pronto para uso imediato.

    Retorna:
      {
        'score': int (0=Real, 100=IA),
        'failed': bool,
        'details': {
            'metrics': {...},
            'model_info': {...}
        },
        'findings': [str]
      }
    """
    if not _try_load_hf_model():
        return {
            'score': 50,
            'failed': True,
            'details': {
                'metrics': {},
                'error': _hf_load_error or 'Modelo Hugging Face indisponível',
                'status': 'offline',
                'help': 'Instale com: pip install transformers'
            },
            'findings': [
                'Modelo pré-treinado Hugging Face não carregado. '
                'Execute: pip install transformers'
            ]
        }

    try:
        import torch

        # Carregar imagem tratando paleta e transparência
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA', 'P', 'PA'):
                img = img.convert('RGBA')
            img_rgb = img.convert('RGB')

        # Pré-processamento
        inputs = _hf_processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(_hf_device) for k, v in inputs.items()}

        # Inferência
        with torch.no_grad():
            outputs = _hf_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Mapeamento de labels do modelo (id2label)
        id2label = _hf_model.config.id2label
        ai_prob = 0.5
        real_prob = 0.5

        for idx, label_name in id2label.items():
            prob_val = float(probs[idx].item())
            lbl = label_name.lower().strip()

            # Mapeia labels como 'artificial', 'ai', 'fake', 'synthetic'
            if any(term in lbl for term in ['art', 'ai', 'fake', 'synth', 'gen']):
                ai_prob = prob_val
            # Mapeia labels como 'human', 'real', 'authentic', 'natural'
            elif any(term in lbl for term in ['hum', 'real', 'auth', 'nat']):
                real_prob = prob_val

        # Se o modelo só mapeou uma das classes explicitamente, calcula a complementar
        if abs((ai_prob + real_prob) - 1.0) > 0.05:
            real_prob = 1.0 - ai_prob

        # Pontuação final (0 a 100)
        score = int(round(ai_prob * 100.0))
        score = min(max(score, 0), 100)

        # Certeza da classificação
        diff = abs(ai_prob - real_prob)
        if diff > 0.5:
            certainty = 'Alta'
        elif diff > 0.2:
            certainty = 'Média'
        else:
            certainty = 'Baixa'

        # Nome do modelo utilizado
        model_name = getattr(_hf_model.config, '_name_or_path', DEFAULT_HF_MODEL)

        # Achados técnicos explicativos
        findings = []
        if score >= 75:
            findings.append(
                f"Rede neural especialista ({model_name}) identificou fortes assinaturas e texturas sintéticas de IA "
                f"com probabilidade de {ai_prob:.1%}."
            )
        elif score <= 25:
            findings.append(
                f"Rede neural especialista ({model_name}) classificou a imagem como fotografia autêntica "
                f"com probabilidade de {real_prob:.1%}."
            )
        else:
            findings.append(
                f"Rede neural especialista ({model_name}) indicou distribuição mista "
                f"(Probabilidade IA: {ai_prob:.1%} / Real: {real_prob:.1%})."
            )

        return {
            'score': score,
            'failed': False,
            'details': {
                'metrics': {
                    'probabilidade_ai': round(ai_prob, 4),
                    'probabilidade_real': round(real_prob, 4),
                    'certeza_classificacao': certainty,
                    'modelo_pretreinado': model_name,
                    'arquitetura': _hf_model.config.model_type if hasattr(_hf_model.config, 'model_type') else 'Vision Transformer'
                },
                'model_info': {
                    'model_id': model_name,
                    'num_labels': len(id2label),
                    'labels': id2label
                }
            },
            'findings': findings
        }

    except Exception as e:
        return {
            'score': 50,
            'failed': True,
            'details': {
                'metrics': {},
                'error': str(e)
            },
            'findings': [f'Erro na análise com modelo pré-treinado: {str(e)}']
        }
