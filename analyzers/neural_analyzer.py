import os
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level singletons — os modelos pré-treinados são carregados uma vez.
# ---------------------------------------------------------------------------
_engines = {
    'engine_1': {
        'candidates': [
            "Organika/sdxl-detector",             # ResNet-50 fine-tuned SDXL, labels: REAL / FAKE
            "Smogy/SMOGY-Ai-images-detector",     # Evolução SDXL detector, labels: AI / REAL
            "cmckinley/deepfake-detection-model", # CLIP fine-tuned
        ],
        'role': 'Especialista em Difusão (SDXL / Latent Artifacts)',
        'processor': None,
        'model': None,
        'model_id': None,
        'arch': None,
        'loaded': False,
        'error': None
    },
    'engine_2': {
        'candidates': [
            "umm-maybe/AI-image-detector",        # ViT Base fine-tuned, labels: artificial / human
            "Nahrawy/AIorNot",                    # Swin-Tiny Transformer, labels: ai / real
            "capcheck/ai-image-detection",        # ViT CIFAKE, labels: REAL / FAKE
        ],
        'role': 'Especialista em Atenção Global (Vision Transformer / Swin)',
        'processor': None,
        'model': None,
        'model_id': None,
        'arch': None,
        'loaded': False,
        'error': None
    }
}

_hf_device = None
_initialization_done = False

# Mapeamento explícito de labels conhecidos → classe IA ou REAL
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
    usando o dicionário de labels do modelo e o mapeamento explícito.
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
        labels = list(id2label.values())
        if len(labels) == 2:
            ai_prob = float(probs_tensor[max(id2label.keys())].item())
            real_prob = 1.0 - ai_prob
        else:
            ai_prob = 0.5
            real_prob = 0.5

    return float(ai_prob), float(real_prob)


def _load_engine(engine_key: str):
    """
    Carrega o melhor modelo disponível para um dos slots de motor neural.
    """
    global _hf_device
    eng = _engines[engine_key]

    if eng['loaded']:
        return eng['model'] is not None

    eng['loaded'] = True

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        if _hf_device is None:
            _hf_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for model_id in eng['candidates']:
            try:
                print(f"[Spectra Neural] [{eng['role']}] Carregando modelo: {model_id} ...")
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModelForImageClassification.from_pretrained(model_id)
                model.to(_hf_device)
                model.eval()

                # Valida se os labels são compreensíveis
                id2label = model.config.id2label
                labels_lower = {str(v).lower().strip() for v in id2label.values()}
                known = labels_lower & (LABEL_MAP_AI | LABEL_MAP_REAL)

                if len(known) == 0:
                    print(f"[Spectra Neural] Modelo {model_id} tem labels desconhecidos: {labels_lower}. Pulando...")
                    continue

                eng['processor'] = processor
                eng['model'] = model
                eng['model_id'] = model_id
                eng['arch'] = getattr(model.config, 'model_type', 'transformer/cnn')
                print(f"[Spectra Neural] [{eng['role']}] Carregado com sucesso: {model_id} ({eng['arch']})")
                return True

            except Exception as e:
                print(f"[Spectra Neural] Falha ao carregar {model_id}: {e}")
                continue

        eng['error'] = f"Nenhum candidato para {eng['role']} pôde ser carregado."
        return False

    except ImportError:
        eng['error'] = "A biblioteca 'transformers' não está instalada."
        return False
    except Exception as e:
        eng['error'] = str(e)
        return False


def _init_neural_engines():
    """Garante a inicialização de ambos os motores neurais."""
    global _initialization_done
    if _initialization_done:
        return
    _initialization_done = True
    _load_engine('engine_1')
    _load_engine('engine_2')


def _run_single_engine(engine_key: str, img_rgb):
    """Executa inferência em um único motor neural e retorna métricas estruturadas."""
    eng = _engines[engine_key]
    if eng['model'] is None:
        _load_engine(engine_key)

    if eng['model'] is None:
        return None

    import torch

    inputs = eng['processor'](images=img_rgb, return_tensors="pt")
    inputs = {k: v.to(_hf_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = eng['model'](**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    id2label = eng['model'].config.id2label
    ai_prob, real_prob = _resolve_probs(id2label, probs)

    score = int(round(min(max(ai_prob * 100.0, 0), 100)))
    diff = abs(ai_prob - real_prob)
    certainty = 'Alta' if diff > 0.5 else ('Média' if diff > 0.2 else 'Baixa')
    model_short = eng['model_id'].split('/')[-1] if eng['model_id'] else 'Neural'

    return {
        'role': eng['role'],
        'model_id': eng['model_id'],
        'model_short': model_short,
        'architecture': eng['arch'],
        'ai_prob': ai_prob,
        'real_prob': real_prob,
        'score': score,
        'certainty': certainty,
        'id2label': {str(k): str(v) for k, v in id2label.items()}
    }


def analyze_neural(image_path: str) -> dict:
    """
    Análise Forense Neural Dual-Engine via Modelos Especialistas Hugging Face.

    Executa inferência paralela/conjunta em dois modelos com arquiteturas distintas:
      1. Motor 1: Especialista em Difusão (SDXL / CNN / ResNet)
      2. Motor 2: Especialista em Atenção Global (Vision Transformer / Swin)

    Realiza fusão por ensemble e calcula concordância/divergência forense.
    """
    _init_neural_engines()

    has_e1 = _engines['engine_1']['model'] is not None
    has_e2 = _engines['engine_2']['model'] is not None

    if not has_e1 and not has_e2:
        err_msg = (
            _engines['engine_1'].get('error') or
            _engines['engine_2'].get('error') or
            'Nenhum modelo neural pôde ser carregado.'
        )
        return {
            'score': 50,
            'failed': True,
            'details': {
                'metrics': {},
                'error': err_msg,
                'help': 'pip install transformers torch torchvision'
            },
            'findings': [f'Detector neural não disponível: {err_msg}']
        }

    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA', 'P', 'PA'):
                img = img.convert('RGBA')
            img_rgb = img.convert('RGB')

        res1 = _run_single_engine('engine_1', img_rgb) if has_e1 else None
        res2 = _run_single_engine('engine_2', img_rgb) if has_e2 else None

        # Caso apenas um dos motores esteja ativo
        if res1 is not None and res2 is None:
            score = res1['score']
            ai_prob = res1['ai_prob']
            real_prob = res1['real_prob']
            findings = [
                f"Motor 1 ({res1['model_short']}): {res1['ai_prob']:.1%} IA / {res1['real_prob']:.1%} Real "
                f"(Certeza: {res1['certainty']}). Motor 2 offline."
            ]
            return {
                'score': score,
                'failed': False,
                'details': {
                    'metrics': {
                        'probabilidade_ai': round(ai_prob, 4),
                        'probabilidade_real': round(real_prob, 4),
                        'certeza_classificacao': res1['certainty'],
                        'modelo_pretreinado': res1['model_id'],
                        'arquitetura': res1['architecture'],
                        'modo_operacao': 'Single Engine (Motor 1)'
                    },
                    'engines': {'engine_1': res1}
                },
                'findings': findings
            }

        if res2 is not None and res1 is None:
            score = res2['score']
            ai_prob = res2['ai_prob']
            real_prob = res2['real_prob']
            findings = [
                f"Motor 2 ({res2['model_short']}): {res2['ai_prob']:.1%} IA / {res2['real_prob']:.1%} Real "
                f"(Certeza: {res2['certainty']}). Motor 1 offline."
            ]
            return {
                'score': score,
                'failed': False,
                'details': {
                    'metrics': {
                        'probabilidade_ai': round(ai_prob, 4),
                        'probabilidade_real': round(real_prob, 4),
                        'certeza_classificacao': res2['certainty'],
                        'modelo_pretreinado': res2['model_id'],
                        'arquitetura': res2['architecture'],
                        'modo_operacao': 'Single Engine (Motor 2)'
                    },
                    'engines': {'engine_2': res2}
                },
                'findings': findings
            }

        # --- Dual Engine Ensemble ---
        w1, w2 = 0.50, 0.50
        # Ajuste fino de pesos baseado em certeza individual
        if res1['certainty'] == 'Alta' and res2['certainty'] != 'Alta':
            w1, w2 = 0.60, 0.40
        elif res2['certainty'] == 'Alta' and res1['certainty'] != 'Alta':
            w1, w2 = 0.40, 0.60

        ai_prob_comb = (w1 * res1['ai_prob']) + (w2 * res2['ai_prob'])
        real_prob_comb = (w1 * res1['real_prob']) + (w2 * res2['real_prob'])
        score = int(round(min(max(ai_prob_comb * 100.0, 0), 100)))

        diff_comb = abs(ai_prob_comb - real_prob_comb)
        certainty_comb = 'Alta' if diff_comb > 0.5 else ('Média' if diff_comb > 0.2 else 'Baixa')

        # Determinação de Concordância
        s1, s2 = res1['score'], res2['score']
        if s1 >= 60 and s2 >= 60:
            concordance = 'Consenso IA (Ambos Modelos Detectaram Síntese)'
            concordance_code = 'consensus_ai'
        elif s1 <= 40 and s2 <= 40:
            concordance = 'Consenso Autêntico (Ambos Modelos Confirmam Foto Real)'
            concordance_code = 'consensus_real'
        elif (s1 >= 60 and s2 <= 40) or (s2 >= 60 and s1 <= 40):
            concordance = 'Divergência Forense (Discrepância entre Difusão e ViT)'
            concordance_code = 'divergence'
        else:
            concordance = 'Concordância Parcial / Sinais Moderados'
            concordance_code = 'partial'

        findings = []
        findings.append(
            f"Motor 1 - Difusão ({res1['model_short']}): {res1['ai_prob']:.1%} IA "
            f"(certeza: {res1['certainty']})"
        )
        findings.append(
            f"Motor 2 - ViT Global ({res2['model_short']}): {res2['ai_prob']:.1%} IA "
            f"(certeza: {res2['certainty']})"
        )

        if concordance_code == 'consensus_ai':
            findings.append(f"Dupla Validação: Consenso total entre modelos ({score}% de probabilidade combinada).")
        elif concordance_code == 'consensus_real':
            findings.append("Dupla Validação: Ambos os modelos classificaram a captura como autêntica.")
        elif concordance_code == 'divergence':
            findings.append(
                f"Alerta Forense: Divergência entre detecção de difusão ({s1}%) e análise de atenção global ({s2}%)."
            )

        return {
            'score': score,
            'failed': False,
            'details': {
                'metrics': {
                    'probabilidade_ai': round(ai_prob_comb, 4),
                    'probabilidade_real': round(real_prob_comb, 4),
                    'certeza_classificacao': certainty_comb,
                    'concordancia_neural': concordance,
                    'motor_1_modelo': res1['model_id'],
                    'motor_1_score': res1['score'],
                    'motor_1_prob_ai': round(res1['ai_prob'], 4),
                    'motor_2_modelo': res2['model_id'],
                    'motor_2_score': res2['score'],
                    'motor_2_prob_ai': round(res2['ai_prob'], 4),
                    'arquitetura_combinada': f"{res1['architecture']} + {res2['architecture']}"
                },
                'engines': {
                    'engine_1': {
                        'role': res1['role'],
                        'model_id': res1['model_id'],
                        'model_short': res1['model_short'],
                        'architecture': res1['architecture'],
                        'score': res1['score'],
                        'ai_prob': round(res1['ai_prob'], 4),
                        'real_prob': round(res1['real_prob'], 4),
                        'certainty': res1['certainty']
                    },
                    'engine_2': {
                        'role': res2['role'],
                        'model_id': res2['model_id'],
                        'model_short': res2['model_short'],
                        'architecture': res2['architecture'],
                        'score': res2['score'],
                        'ai_prob': round(res2['ai_prob'], 4),
                        'real_prob': round(res2['real_prob'], 4),
                        'certainty': res2['certainty']
                    }
                },
                'concordance_code': concordance_code,
                'concordance_text': concordance
            },
            'findings': findings
        }

    except Exception as e:
        return {
            'score': 50,
            'failed': True,
            'details': {'metrics': {}, 'error': str(e)},
            'findings': [f'Erro na inferência neural dual: {str(e)}']
        }
