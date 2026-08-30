import os
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level singleton — o modelo pré-treinado é carregado uma única vez.
# ---------------------------------------------------------------------------
_hf_processor = None
_hf_model = None
_hf_device = None
_hf_loaded = False
_hf_load_error = None
_loaded_model_id = None

# Modelos em ordem de prioridade — escolhemos modelos com labels claros e
# treinados em geradores modernos (Midjourney, SD, SDXL, Flux, DALL-E 3).
CANDIDATE_MODELS = [
    "Organika/sdxl-detector",                    # ResNet-50 fine-tuned, labels: REAL / FAKE
    "Nahrawy/AIorNot",                            # ViT fine-tuned, labels: ai / real
    "cmckinley/deepfake-detection-model",         # CLIP fine-tuned, detecta geração difusa
]

# Mapeamento explícito de labels conhecidos → classe IA ou REAL
# Usado para garantir que a leitura seja correta independentemente do modelo.
LABEL_MAP_AI = {
    'fake', 'artificial', 'ai', 'ai-generated', 'generated',
    'synthetic', 'synthetic_image', 'aiart', 'diffusion', '1'
}
LABEL_MAP_REAL = {
    'real', 'human', 'authentic', 'natural', 'photo', 'photograph',
    'human_art', 'not_ai', '0'
}


def _resolve_probs(id2label, probs_tensor) -> tuple[float, float]:
    """
    Mapeia os logits do modelo para (prob_ai, prob_real) de forma robusta,
    usando o dicionário de labels do modelo e o mapeamento explícito acima.
    """
    ai_prob = None
    real_prob = None

    for idx, label in id2label.items():
        lbl = str(label).lower().strip().replace(' ', '_').replace('-', '_')
        prob_val = float(probs_tensor[idx].item())

        if lbl in LABEL_MAP_AI:
            ai_prob = prob_val
        elif lbl in LABEL_MAP_REAL:
            real_prob = prob_val

    # Se apenas uma foi mapeada, calcula a complementar
    if ai_prob is not None and real_prob is None:
        real_prob = 1.0 - ai_prob
    elif real_prob is not None and ai_prob is None:
        ai_prob = 1.0 - real_prob
    elif ai_prob is None and real_prob is None:
        # Fallback: assume índice 0 = primeira classe, pega a de maior prob
        # e tenta inferir pela posição
        labels = list(id2label.values())
        if len(labels) == 2:
            # tenta a classe de índice maior como IA (convenção comum)
            ai_prob = float(probs_tensor[max(id2label.keys())].item())
            real_prob = 1.0 - ai_prob
        else:
            ai_prob = 0.5
            real_prob = 0.5

    return float(ai_prob), float(real_prob)


def _try_load_hf_model():
    """
    Carrega o melhor modelo de detecção de IA disponível do Hugging Face.
    Tenta os candidatos em sequência até encontrar um que carregue com sucesso.
    """
    global _hf_processor, _hf_model, _hf_device, _hf_loaded, _hf_load_error, _loaded_model_id

    if _hf_loaded:
        return _hf_model is not None

    _hf_loaded = True

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        _hf_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for model_id in CANDIDATE_MODELS:
            try:
                print(f"[Spectra Neural] Carregando modelo: {model_id} ...")
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModelForImageClassification.from_pretrained(model_id)
                model.to(_hf_device)
                model.eval()

                # Verifica se os labels do modelo são mapeáveis
                id2label = model.config.id2label
                labels_lower = {str(v).lower().strip() for v in id2label.values()}
                known = labels_lower & (LABEL_MAP_AI | LABEL_MAP_REAL)

                if len(known) == 0:
                    print(f"[Spectra Neural] Modelo {model_id} tem labels desconhecidos: {labels_lower}. Pulando...")
                    continue

                _hf_processor = processor
                _hf_model = model
                _loaded_model_id = model_id
                print(f"[Spectra Neural] Modelo carregado com sucesso: {model_id}")
                print(f"[Spectra Neural] Labels: {id2label}")
                return True

            except Exception as e:
                print(f"[Spectra Neural] Falha ao carregar {model_id}: {e}")
                continue

        _hf_load_error = (
            "Nenhum modelo de detecção de IA pôde ser carregado. "
            "Verifique sua conexão com a internet e a instalação de 'transformers' e 'torch'."
        )
        return False

    except ImportError:
        _hf_load_error = (
            "A biblioteca 'transformers' não está instalada. "
            "Execute: pip install transformers torch torchvision"
        )
        return False
    except Exception as e:
        _hf_load_error = str(e)
        return False


def analyze_neural(image_path: str) -> dict:
    """
    Análise de Deep Learning via modelo especialista pré-treinado (Hugging Face).

    Detecta imagens sintéticas de IA (Stable Diffusion, Midjourney, SDXL, etc.)
    versus fotografias reais, usando modelos de visão computacional treinados
    especificamente para este problema.

    NÃO requer treinamento local — os pesos são baixados automaticamente.
    """
    if not _try_load_hf_model():
        return {
            'score': 50,
            'failed': True,
            'details': {
                'metrics': {},
                'error': _hf_load_error or 'Modelo não disponível',
                'help': 'pip install transformers torch torchvision'
            },
            'findings': [
                f'Modelo neural não disponível: {_hf_load_error}'
            ]
        }

    try:
        import torch

        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA', 'P', 'PA'):
                img = img.convert('RGBA')
            img_rgb = img.convert('RGB')

        inputs = _hf_processor(images=img_rgb, return_tensors="pt")
        inputs = {k: v.to(_hf_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _hf_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        id2label = _hf_model.config.id2label
        ai_prob, real_prob = _resolve_probs(id2label, probs)

        score = int(round(min(max(ai_prob * 100.0, 0), 100)))

        diff = abs(ai_prob - real_prob)
        certainty = 'Alta' if diff > 0.5 else ('Média' if diff > 0.2 else 'Baixa')

        model_short = _loaded_model_id.split('/')[-1] if _loaded_model_id else 'Neural'

        findings = []
        if score >= 75:
            findings.append(
                f"Detector Neural ({model_short}) identificou fortes assinaturas sintéticas de IA "
                f"com probabilidade de {ai_prob:.1%} (certeza: {certainty})."
            )
        elif score <= 25:
            findings.append(
                f"Detector Neural ({model_short}) classificou como fotografia autêntica "
                f"com probabilidade de {real_prob:.1%} (certeza: {certainty})."
            )
        else:
            findings.append(
                f"Detector Neural ({model_short}) encontrou sinais mistos "
                f"(IA: {ai_prob:.1%} / Real: {real_prob:.1%})."
            )

        return {
            'score': score,
            'failed': False,
            'details': {
                'metrics': {
                    'probabilidade_ai': round(ai_prob, 4),
                    'probabilidade_real': round(real_prob, 4),
                    'certeza_classificacao': certainty,
                    'modelo_pretreinado': _loaded_model_id or model_short,
                    'arquitetura': getattr(_hf_model.config, 'model_type', 'unknown')
                },
                'model_info': {
                    'model_id': _loaded_model_id,
                    'labels': {str(k): str(v) for k, v in id2label.items()}
                }
            },
            'findings': findings
        }

    except Exception as e:
        return {
            'score': 50,
            'failed': True,
            'details': {'metrics': {}, 'error': str(e)},
            'findings': [f'Erro na inferência neural: {str(e)}']
        }
