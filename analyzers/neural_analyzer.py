import os
import json
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level singleton — O modelo EfficientNetV2 fine-tuned é carregado
# uma única vez e reutilizado em todas as requisições subsequentes.
# ---------------------------------------------------------------------------
_neural_model = None
_neural_transform = None
_neural_device = None
_neural_meta = None
_neural_loaded = False
_neural_error = None

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'spectra_transfer_model.pt')
META_PATH = os.path.join(MODELS_DIR, 'transfer_model_meta.json')


def _build_architecture(backbone_name: str, num_classes: int = 2):
    """Reconstrói a arquitetura do modelo correspondente ao checkpoint salvo."""
    import torch.nn as nn
    import torchvision.models as models

    if backbone_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )
    elif backbone_name == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )
    elif backbone_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )
    else:
        # Fallback para efficientnet_v2_s
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )

    return model


def _try_load_neural_model():
    """
    Tenta carregar o modelo de Transfer Learning treinado.
    Retorna True se carregado com sucesso, False caso contrário.
    """
    global _neural_model, _neural_transform, _neural_device, _neural_meta, _neural_loaded, _neural_error

    if _neural_loaded:
        return _neural_model is not None

    _neural_loaded = True

    try:
        import torch
        import torchvision.transforms as transforms

        _neural_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(MODEL_PATH):
            _neural_error = (
                f"Modelo treinado não encontrado em '{MODEL_PATH}'. "
                "Execute 'python train_transfer_learning.py' para treinar com seu dataset."
            )
            return False

        checkpoint = torch.load(MODEL_PATH, map_location=_neural_device)
        backbone_name = checkpoint.get('backbone', 'efficientnet_v2_s')

        model = _build_architecture(backbone_name, num_classes=2)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(_neural_device)
        model.eval()

        _neural_model = model

        _neural_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, 'r', encoding='utf-8') as f:
                    _neural_meta = json.load(f)
            except Exception:
                _neural_meta = {}
        else:
            _neural_meta = {
                'backbone': backbone_name,
                'accuracy': checkpoint.get('accuracy', 0.0),
                'trained_at': checkpoint.get('timestamp', 'Desconhecido')
            }

        return True

    except Exception as e:
        _neural_error = str(e)
        return False


def analyze_neural(image_path: str) -> dict:
    """
    Analisador de Deep Learning via Transfer Learning (EfficientNetV2 fine-tuned).

    Avalia a imagem através de representações convolucionais profundas
    treinadas para distinguir fotos reais de imagens geradas por IA.

    Retorna:
      {
        'score': int (0=Real, 100=IA),
        'failed': bool,
        'details': {
            'metrics': {...},
            'backbone': str,
            'model_info': {...}
        },
        'findings': [str]
      }
    """
    if not _try_load_neural_model():
        return {
            'score': 50,
            'failed': True,
            'details': {
                'metrics': {},
                'error': _neural_error or 'Modelo Transfer Learning não disponível',
                'status': 'untrained',
                'help': 'Execute: python train_transfer_learning.py'
            },
            'findings': [
                'Modelo de Transfer Learning não treinado ou indisponível. '
                'Execute: python train_transfer_learning.py'
            ]
        }

    try:
        import torch

        # Carregar e pré-processar imagem
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA', 'P', 'PA'):
                img = img.convert('RGBA')
            img_rgb = img.convert('RGB')
            tensor = _neural_transform(img_rgb).unsqueeze(0).to(_neural_device)

        # Inferência
        with torch.no_grad():
            outputs = _neural_model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            real_prob = float(probs[0].item())
            ai_prob = float(probs[1].item())

        # Pontuação 0 a 100
        score = int(round(ai_prob * 100.0))
        score = min(max(score, 0), 100)

        # Achados e explicações forenses
        findings = []
        backbone_name = _neural_meta.get('backbone', 'EfficientNetV2') if _neural_meta else 'EfficientNetV2'
        acc_text = f" (Acurácia de validação do modelo: {_neural_meta.get('accuracy', 0):.1%})" if _neural_meta and 'accuracy' in _neural_meta else ""

        if score >= 75:
            findings.append(
                f"Rede neural profunda ({backbone_name}) identificou fortes padrões e assinaturas sintéticas de IA "
                f"com probabilidade de {ai_prob:.1%}{acc_text}."
            )
        elif score <= 25:
            findings.append(
                f"Rede neural profunda ({backbone_name}) detectou características naturais e consistentes com foto real "
                f"com probabilidade de {real_prob:.1%}{acc_text}."
            )
        else:
            findings.append(
                f"Rede neural profunda ({backbone_name}) indicou padrões visuais mistos "
                f"(Probabilidade IA: {ai_prob:.1%} / Real: {real_prob:.1%})."
            )

        # Nível de certeza
        diff = abs(ai_prob - real_prob)
        certainty = 'Alta' if diff > 0.5 else ('Média' if diff > 0.2 else 'Baixa')

        return {
            'score': score,
            'failed': False,
            'details': {
                'metrics': {
                    'probabilidade_ai': round(ai_prob, 4),
                    'probabilidade_real': round(real_prob, 4),
                    'certeza_classificacao': certainty,
                    'backbone': backbone_name,
                    'treinado_em': _neural_meta.get('trained_at', 'Recente') if _neural_meta else 'N/A'
                },
                'model_meta': _neural_meta or {}
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
            'findings': [f'Erro durante a inferência neural: {str(e)}']
        }
