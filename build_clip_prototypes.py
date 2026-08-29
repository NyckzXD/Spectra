#!/usr/bin/env python3
"""
build_clip_prototypes.py — Treina os protótipos CLIP a partir do dataset rotulado.

O que faz:
  1. Lê todas as imagens de dataset/Real/ e dataset/AI/
  2. Extrai o embedding CLIP ViT-B/32 de cada imagem
  3. Calcula o centroide (média dos embeddings) de cada classe
  4. Salva em models/clip_prototypes.npz

Como usar:
  pip install open-clip-torch      (primeira vez)
  python build_clip_prototypes.py  (rode após adicionar imagens ao dataset)

Quanto mais imagens e mais variadas as fontes (Midjourney, SDXL, Flux,
DALL-E, fotos de câmeras diferentes), mais robusto fica o protótipo.

Recomendação: rodar novamente após expandir o dataset.
"""
import os
import sys
import numpy as np
from PIL import Image

# Paths
_HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(_HERE, 'dataset', 'Real')
AI_DIR = os.path.join(_HERE, 'dataset', 'AI')
MODELS_DIR = os.path.join(_HERE, 'models')
PROTOTYPES_PATH = os.path.join(MODELS_DIR, 'clip_prototypes.npz')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


def list_images(directory: str):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Diretório não encontrado: {directory}")
    paths = []
    for name in sorted(os.listdir(directory)):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            paths.append(os.path.join(directory, name))
    return paths


def extract_embeddings(paths, model, transform, label: str):
    """Extract CLIP image embeddings for a list of image paths."""
    import torch

    embeddings = []
    skipped = 0
    for i, path in enumerate(paths, 1):
        print(f"  [{i}/{len(paths)}] {os.path.basename(path)}", end='', flush=True)
        try:
            img = Image.open(path).convert('RGB')
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                feat = model.encode_image(tensor).float()
                feat = feat / feat.norm(dim=-1, keepdim=True)
            embeddings.append(feat[0].cpu().numpy().astype(np.float64))
            print(' ✓')
        except Exception as e:
            print(f' ✗ (skipped: {e})')
            skipped += 1

    if skipped:
        print(f"  Aviso: {skipped} imagem(ns) ignorada(s).")
    return np.array(embeddings) if embeddings else np.zeros((0, 512))


def compute_train_accuracy(real_embeddings, ai_embeddings, real_centroid, ai_centroid):
    """Quick in-sample validation of the prototypes."""
    real_c = real_centroid / (np.linalg.norm(real_centroid) + 1e-8)
    ai_c = ai_centroid / (np.linalg.norm(ai_centroid) + 1e-8)

    correct, total = 0, 0

    for emb in real_embeddings:
        if np.dot(emb, real_c) >= np.dot(emb, ai_c):
            correct += 1
        total += 1

    for emb in ai_embeddings:
        if np.dot(emb, ai_c) > np.dot(emb, real_c):
            correct += 1
        total += 1

    return correct, total


def main():
    # --- Check open_clip ---
    try:
        import open_clip
        import torch
    except ImportError:
        print("ERRO: open-clip-torch não está instalado.")
        print("Instale com: pip install open-clip-torch")
        sys.exit(1)

    # --- Load model ---
    print("Carregando CLIP ViT-B/32 (openai)...")
    print("(Primeira execução baixa ~350MB e cacheia em ~/.cache/clip)")
    model, _, transform = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    model.eval()
    print("Modelo carregado.\n")

    # --- List dataset ---
    try:
        real_paths = list_images(REAL_DIR)
        ai_paths = list_images(AI_DIR)
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        sys.exit(1)

    print(f"Dataset: {len(real_paths)} imagens reais, {len(ai_paths)} imagens de IA")

    if len(real_paths) < 3 or len(ai_paths) < 3:
        print("AVISO: dataset muito pequeno (<3 por classe).")
        print("Os protótipos serão instáveis. Recomendado: 30+ por classe.")

    # --- Extract embeddings ---
    print("\nExtraindo embeddings das imagens reais...")
    real_embeddings = extract_embeddings(real_paths, model, transform, 'real')

    print("\nExtraindo embeddings das imagens de IA...")
    ai_embeddings = extract_embeddings(ai_paths, model, transform, 'ai')

    if len(real_embeddings) == 0 or len(ai_embeddings) == 0:
        print("\nERRO: não foi possível extrair embeddings de pelo menos uma classe.")
        sys.exit(1)

    # --- Compute centroids ---
    real_centroid = real_embeddings.mean(axis=0)
    ai_centroid = ai_embeddings.mean(axis=0)

    # --- Save ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.savez(
        PROTOTYPES_PATH,
        real_centroid=real_centroid.astype(np.float32),
        ai_centroid=ai_centroid.astype(np.float32),
        n_real=np.int64(len(real_embeddings)),
        n_ai=np.int64(len(ai_embeddings)),
    )

    # --- Report ---
    correct, total = compute_train_accuracy(real_embeddings, ai_embeddings, real_centroid, ai_centroid)
    accuracy = correct / total if total else 0.0

    print(f"\n{'='*50}")
    print(f"Protótipos salvos em: {PROTOTYPES_PATH}")
    print(f"  Centroide real: {len(real_embeddings)} imagens")
    print(f"  Centroide IA:   {len(ai_embeddings)} imagens")
    print(f"\nAcurácia in-sample: {correct}/{total} = {accuracy:.1%}")
    print("(Nota: acurácia in-sample superestima a performance real.")
    print(" Após expandir o dataset, rodar novamente para retreinar.)")
    print(f"{'='*50}")
    print("\nPróximos passos:")
    print("  1. Reinicie o app.py para carregar os novos protótipos")
    print("  2. Adicione mais imagens ao dataset para melhorar a acurácia")
    print("  3. Rode novamente após adicionar imagens: python build_clip_prototypes.py")


if __name__ == '__main__':
    main()
