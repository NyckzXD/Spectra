#!/usr/bin/env python3
"""
calibrate.py — Ferramenta de calibração empírica para o Spectra.

PROBLEMA QUE ESTE SCRIPT RESOLVE
---------------------------------
Todos os thresholds usados nos 6 analisadores (ex: `poisson_corr > 0.30`,
`spectral_slope_alpha entre 0.85 e 1.30`, `edge_cv > 0.65`) foram definidos
manualmente, sem validação contra imagens reais. Este script substitui o
"chute" por medição: roda os analisadores num dataset rotulado por você
(fotos reais de um lado, imagens de IA de outro) e responde duas perguntas
para cada métrica interna:

  1. Essa métrica REALMENTE separa real de IA no seu dataset? (AUC)
  2. Se sim, qual é o ponto de corte ótimo? (threshold via estatística de Youden)

Também avalia a acurácia do composite score ATUAL do app.py (matriz de
confusão, precisão, recall) para você ter uma linha de base antes de mexer
em qualquer threshold.

COMO USAR
---------
1. Organize seu dataset em duas pastas:

     dataset/
       real/   <- fotos tiradas por câmera/celular (o mais variado possível:
                   com e sem EXIF, direto da câmera E baixadas do
                   WhatsApp/Instagram/Twitter, para testar robustez real)
       ai/     <- imagens geradas por IA (várias ferramentas: Midjourney,
                   SDXL, Flux, DALL-E, etc., com e sem metadados)

   Quanto mais imagens e mais variadas as fontes, mais confiável o resultado.
   Recomendo pelo menos 30-50 de cada categoria para um sinal minimamente
   estável; 150+ de cada é o ideal para calibração de produção.

2. Rode:

     python3 calibrate.py --real-dir dataset/real --ai-dir dataset/ai \
         --project-root /caminho/para/o/projeto/spectra

   `--project-root` é o diretório que contém `app.py` e o pacote `analyzers/`.

3. O script gera:
     - calibration_report.md   (leitura humana: rankeia métricas por AUC,
                                 sugere thresholds, mostra confusão atual)
     - calibration_data.csv    (todas as métricas extraídas, uma linha por
                                 imagem — útil para você plotar/explorar)
     - calibration_raw.json    (mesmo dado em JSON, para uso programático)

O QUE FAZER COM O RESULTADO
----------------------------
Métricas com AUC próximo de 0.5 não separam nada no seu dataset — o
analisador correspondente pode estar medindo ruído, ou a heurística não é
válida para o tipo de imagem que você está classificando. Métricas com AUC
alto (>0.80) são sinais fortes; use o threshold sugerido para substituir o
valor "chutado" no código do analisador correspondente.
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


# --------------------------------------------------------------------------
# Coleta de dados
# --------------------------------------------------------------------------

def list_images(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Diretório não encontrado: {directory}")
    paths = []
    for name in sorted(os.listdir(directory)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            paths.append(os.path.join(directory, name))
    return paths


def flatten_metrics(analyses, prefix_sep='.'):
    """
    Achata o dicionário de análises em pares chave->valor numéricos, com
    prefixo do analisador (ex: 'noise.poisson_correlation', 'spectral.alpha').
    Ignora valores não-numéricos (strings, listas, base64 de visualização).
    """
    flat = {}
    for analyzer_name, result in analyses.items():
        flat[f'{analyzer_name}{prefix_sep}score'] = result.get('score')
        metrics = result.get('details', {}).get('metrics', {})
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[f'{analyzer_name}{prefix_sep}{metric_name}'] = float(value)
    return flat


def run_analyzers_on_dataset(real_paths, ai_paths, analyze_funcs):
    """
    Roda todos os analisadores em cada imagem do dataset.
    Retorna lista de dicts: {'path', 'label' (1=ai, 0=real), 'analyses', **metrics_achatadas}
    """
    rows = []
    all_paths = [(p, 0) for p in real_paths] + [(p, 1) for p in ai_paths]
    total = len(all_paths)

    for i, (path, label) in enumerate(all_paths, 1):
        print(f"[{i}/{total}] Analisando: {os.path.basename(path)}")
        analyses = {}
        for key, func in analyze_funcs.items():
            try:
                analyses[key] = func(path)
            except Exception as e:
                print(f"  AVISO: {key} falhou em {path}: {e}")
                analyses[key] = {'score': None, 'details': {'metrics': {}}, 'failed': True}

        row = {'path': path, 'label': label}
        row.update(flatten_metrics(analyses))
        row['_analyses'] = analyses
        rows.append(row)

    return rows


# --------------------------------------------------------------------------
# Estatística: AUC via Mann-Whitney U (rank-based) e threshold ótimo (Youden)
# --------------------------------------------------------------------------

def compute_auc(values, labels):
    """
    AUC = probabilidade de que um exemplo positivo (label=1, IA) tenha valor
    maior que um exemplo negativo (label=0, real), estimada por ranking
    (equivalente à estatística U de Mann-Whitney / Wilcoxon).
    Não depende de sklearn.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    ranks = rankdata(values)
    sum_ranks_pos = float(np.sum(ranks[labels == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def best_threshold_youden(values, labels):
    """
    Encontra o threshold que maximiza a estatística J de Youden (TPR - FPR),
    testando cada valor único observado como ponto de corte.
    Retorna (threshold, direção, tpr, fpr, j_statistic).
    'direção' = 'higher_is_ai' se valores altos indicam IA, 'higher_is_real' caso contrário.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    candidates = np.unique(values)
    # também testa pontos médios entre valores consecutivos (thresholds mais "limpos")
    if len(candidates) > 1:
        midpoints = (candidates[:-1] + candidates[1:]) / 2.0
        candidates = np.unique(np.concatenate([candidates, midpoints]))

    best = None
    for direction in ('higher_is_ai', 'higher_is_real'):
        for t in candidates:
            if direction == 'higher_is_ai':
                pred_ai = values >= t
            else:
                pred_ai = values < t

            tp = int(np.sum(pred_ai & (labels == 1)))
            fp = int(np.sum(pred_ai & (labels == 0)))
            fn = int(np.sum((~pred_ai) & (labels == 1)))
            tn = int(np.sum((~pred_ai) & (labels == 0)))

            tpr = tp / n_pos if n_pos else 0.0
            fpr = fp / n_neg if n_neg else 0.0
            j = tpr - fpr

            if best is None or j > best['j']:
                best = {
                    'threshold': float(t), 'direction': direction,
                    'tpr': tpr, 'fpr': fpr, 'j': j,
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                }

    return best


def confusion_matrix_report(scores, labels, threshold=50):
    """Matriz de confusão simples usando um corte fixo (composite score >= threshold => 'IA')."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pred_ai = scores >= threshold

    tp = int(np.sum(pred_ai & (labels == 1)))
    fp = int(np.sum(pred_ai & (labels == 0)))
    fn = int(np.sum((~pred_ai) & (labels == 1)))
    tn = int(np.sum((~pred_ai) & (labels == 0)))

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        'threshold': threshold, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
    }


# --------------------------------------------------------------------------
# Regressão logística simples (gradiente descendente, sem sklearn) —
# alternativa a pesos "chutados" no composite score, aprendida a partir do
# próprio dataset rotulado.
# --------------------------------------------------------------------------

def train_logistic_regression(X, y, epochs=3000, lr=0.5, l2=0.01):
    """
    Regressão logística com features padronizadas (z-score).
    X: matriz (n_amostras, n_features) já com apenas os 6 scores dos analisadores.
    y: rótulos (0=real, 1=ia).
    Retorna (weights, bias, mean, std) para poder aplicar em novas imagens.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

    n, d = Xn.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        z = Xn @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = Xn.T @ (p - y) / n + l2 * w
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, mean, std


def logistic_predict_proba(X, w, b, mean, std):
    X = np.asarray(X, dtype=np.float64)
    Xn = (X - mean) / std
    z = Xn @ w + b
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------
# Relatório
# --------------------------------------------------------------------------

def build_report(rows, current_composite_scores, verdict_func, out_dir):
    labels = [r['label'] for r in rows]

    # --- 1. Ranking de métricas individuais por AUC ---
    metric_names = sorted({
        k for r in rows for k in r.keys()
        if k not in ('path', 'label', '_analyses')
    })

    metric_results = []
    for m in metric_names:
        values = [r.get(m) for r in rows]
        if any(v is None for v in values):
            continue  # métrica ausente em alguma imagem (ex: analisador falhou) -> pula
        auc = compute_auc(values, labels)
        if auc is None:
            continue
        # AUC < 0.5 significa que a direção está invertida; normalizamos para "poder discriminativo"
        discriminative_power = max(auc, 1 - auc)
        bt = best_threshold_youden(values, labels)
        metric_results.append({
            'metric': m, 'auc': auc, 'discriminative_power': discriminative_power,
            'suggested_threshold': bt['threshold'] if bt else None,
            'direction': bt['direction'] if bt else None,
            'tpr_at_threshold': bt['tpr'] if bt else None,
            'fpr_at_threshold': bt['fpr'] if bt else None,
        })

    metric_results.sort(key=lambda x: x['discriminative_power'], reverse=True)

    # --- 2. Matriz de confusão do composite score ATUAL (app.py) ---
    current_cm = confusion_matrix_report(current_composite_scores, labels, threshold=50)

    # --- 3. Matriz de confusão por analisador individual (score bruto >=50) ---
    per_analyzer_cm = {}
    for analyzer in ['metadata', 'noise', 'spectral', 'statistical', 'wavelet', 'artifacts', 'clip']:
        col = f'{analyzer}.score'
        values = [r.get(col) for r in rows]
        if any(v is None for v in values):
            continue
        per_analyzer_cm[analyzer] = confusion_matrix_report(values, labels, threshold=50)
        per_analyzer_cm[analyzer]['auc'] = compute_auc(values, labels)

    # --- 4. Regressão logística aprendida a partir dos 6 scores ---
    analyzer_order = ['metadata', 'noise', 'spectral', 'statistical', 'wavelet', 'artifacts', 'clip']
    X = []
    valid_idx = []
    for i, r in enumerate(rows):
        vec = [r.get(f'{a}.score') for a in analyzer_order]
        if all(v is not None for v in vec):
            X.append(vec)
            valid_idx.append(i)
    y = [labels[i] for i in valid_idx]

    logistic_report = None
    if len(set(y)) == 2 and len(y) >= 10:
        w, b, mean, std = train_logistic_regression(np.array(X), np.array(y))
        proba = logistic_predict_proba(np.array(X), w, b, mean, std)
        pred = (proba >= 0.5).astype(int)
        acc = float(np.mean(pred == np.array(y)))
        logistic_report = {
            'weights': dict(zip(analyzer_order, w.tolist())),
            'bias': float(b),
            'feature_mean': dict(zip(analyzer_order, mean.tolist())),
            'feature_std': dict(zip(analyzer_order, std.tolist())),
            'train_accuracy': acc,
            'n_samples': len(y),
        }

    # --- Escreve CSV ---
    csv_path = os.path.join(out_dir, 'calibration_data.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['path', 'label'] + metric_names
        writer.writerow(header)
        for r in rows:
            writer.writerow([r['path'], r['label']] + [r.get(m, '') for m in metric_names])

    # --- Escreve JSON bruto ---
    json_path = os.path.join(out_dir, 'calibration_raw.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metric_ranking': metric_results,
            'current_composite_confusion': current_cm,
            'per_analyzer_confusion': per_analyzer_cm,
            'logistic_regression': logistic_report,
            'n_real': int(sum(1 for l in labels if l == 0)),
            'n_ai': int(sum(1 for l in labels if l == 1)),
        }, f, indent=2)

    # --- Escreve Markdown legível ---
    md_path = os.path.join(out_dir, 'calibration_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório de Calibração — Spectra\n\n")
        f.write(f"Dataset: {sum(1 for l in labels if l==0)} imagens reais, "
                f"{sum(1 for l in labels if l==1)} imagens de IA.\n\n")

        f.write("## 1. Desempenho do composite score ATUAL (app.py, corte em 50)\n\n")
        f.write(_cm_table(current_cm))
        f.write("\n")
        if current_cm['accuracy'] < 0.75:
            f.write("⚠️ Acurácia abaixo de 75% — indica que os pesos/thresholds atuais "
                    "não estão bem calibrados para este dataset.\n\n")

        f.write("## 2. Desempenho de cada analisador isoladamente (score bruto, corte em 50)\n\n")
        f.write("| Analisador | AUC | Acurácia | Precisão | Recall | F1 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for a, cm in per_analyzer_cm.items():
            f.write(f"| {a} | {cm['auc']:.3f} | {cm['accuracy']:.2%} | "
                    f"{cm['precision']:.2%} | {cm['recall']:.2%} | {cm['f1']:.2%} |\n")
        f.write("\nAUC próximo de 0.50 = o analisador não está discriminando nada no seu "
                "dataset (é ruído). AUC > 0.80 = sinal forte.\n\n")

        f.write("## 3. Ranking de métricas internas por poder discriminativo (AUC)\n\n")
        f.write("Estas são as métricas cruas por trás de cada score — use para recalibrar "
                "os thresholds hardcoded no código de cada analisador.\n\n")
        f.write("| Métrica | AUC | Direção | Threshold sugerido | TPR | FPR |\n")
        f.write("|---|---|---|---|---|---|\n")
        for m in metric_results[:40]:
            direction_label = ("valor alto → IA" if m['direction'] == 'higher_is_ai'
                                else "valor baixo → IA")
            f.write(f"| `{m['metric']}` | {m['auc']:.3f} | {direction_label} | "
                    f"{m['suggested_threshold']:.4f} | {m['tpr_at_threshold']:.2%} | "
                    f"{m['fpr_at_threshold']:.2%} |\n")
        f.write("\n")

        if logistic_report:
            f.write("## 4. Alternativa: pesos aprendidos por regressão logística\n\n")
            f.write(f"Treinada nos 6 scores dos analisadores ({logistic_report['n_samples']} "
                    f"amostras). Acurácia de treino: {logistic_report['train_accuracy']:.2%} "
                    f"(referência, não validação cruzada — dataset pequeno tende a "
                    f"superestimar; use com cautela e idealmente valide em imagens novas).\n\n")
            f.write("| Analisador | Peso aprendido |\n|---|---|\n")
            for a, w in logistic_report['weights'].items():
                sign = "puxa para IA" if w > 0 else "puxa para real"
                f.write(f"| {a} | {w:+.4f} ({sign}) |\n")
            f.write(f"\nBias: {logistic_report['bias']:+.4f}\n\n")
            f.write("Compare esses pesos com os `base_weights` hardcoded em "
                    "`calculate_composite_score` (app.py) — se a ordem de importância for "
                    "muito diferente, é sinal de que a ponderação manual atual não reflete "
                    "o que realmente separa real de IA no seu dataset.\n\n")

        f.write("## Como usar este relatório\n\n")
        f.write("1. Olhe a seção 2 primeiro: se algum analisador tem AUC ~0.5, ele está "
                "adicionando ruído ao composite score, não sinal — considere reduzir seu "
                "peso ou revisar sua lógica.\n"
                "2. Na seção 3, para cada métrica com AUC alto, abra o analisador "
                "correspondente e troque o threshold hardcoded pelo `threshold sugerido`.\n"
                "3. Repita a calibração após as mudanças, idealmente com imagens novas "
                "(não as mesmas usadas para calibrar), para confirmar que a acurácia "
                "realmente melhorou e não foi só overfitting no dataset de calibração.\n")

    return md_path, csv_path, json_path


def _cm_table(cm):
    return (
        f"| | Previsto: Real | Previsto: IA |\n"
        f"|---|---|---|\n"
        f"| **Real de verdade** | {cm['tn']} (TN) | {cm['fp']} (FP) |\n"
        f"| **IA de verdade** | {cm['fn']} (FN) | {cm['tp']} (TP) |\n\n"
        f"Acurácia: {cm['accuracy']:.2%} | Precisão: {cm['precision']:.2%} | "
        f"Recall: {cm['recall']:.2%} | F1: {cm['f1']:.2%}\n"
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibração empírica do Spectra")
    parser.add_argument('--real-dir', required=True, help='Pasta com imagens reais rotuladas')
    parser.add_argument('--ai-dir', required=True, help='Pasta com imagens de IA rotuladas')
    parser.add_argument('--project-root', required=True,
                         help='Caminho para o diretório do projeto (contém app.py e analyzers/)')
    parser.add_argument('--output-dir', default='.', help='Onde salvar os relatórios')
    args = parser.parse_args()

    sys.path.insert(0, args.project_root)
    try:
        from analyzers.metadata_analyzer import analyze_metadata
        from analyzers.wavelet_analyzer import analyze_wavelet
        from analyzers.clip_analyzer import analyze_clip
        from analyzers.spectral_analyzer import analyze_spectral
        from analyzers.noise_analyzer import analyze_noise
        from analyzers.statistical_analyzer import analyze_statistical
        from analyzers.artifact_analyzer import analyze_artifacts
        from app import calculate_composite_score, get_verdict
    except ImportError as e:
        print(f"ERRO: não consegui importar o projeto em '{args.project_root}'.\n"
              f"Confirme que a pasta contém app.py e o pacote analyzers/.\nDetalhe: {e}")
        sys.exit(1)

    analyze_funcs = {
        'metadata': analyze_metadata,
        'wavelet': analyze_wavelet,
        'noise': analyze_noise,
        'spectral': analyze_spectral,
        'statistical': analyze_statistical,
        'artifacts': analyze_artifacts,
        'clip': analyze_clip,
    }

    real_paths = list_images(args.real_dir)
    ai_paths = list_images(args.ai_dir)
    print(f"Encontradas {len(real_paths)} imagens reais e {len(ai_paths)} imagens de IA.")
    if len(real_paths) < 5 or len(ai_paths) < 5:
        print("AVISO: dataset muito pequeno (<5 por classe). O resultado será instável — "
              "trate como indicativo, não conclusivo.")

    start = time.time()
    rows = run_analyzers_on_dataset(real_paths, ai_paths, analyze_funcs)
    print(f"Análise concluída em {time.time() - start:.1f}s")

    composite_scores = []
    for r in rows:
        try:
            composite_scores.append(calculate_composite_score(r['_analyses']))
        except Exception:
            traceback.print_exc()
            composite_scores.append(50)

    os.makedirs(args.output_dir, exist_ok=True)
    md_path, csv_path, json_path = build_report(rows, composite_scores, get_verdict, args.output_dir)

    print("\n=== Relatório gerado ===")
    print(f"  {md_path}")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == '__main__':
    main()