#!/usr/bin/env python3
"""
train_transfer_learning.py — Treina um modelo de Transfer Learning para detecção de IA.

Utiliza a arquitetura EfficientNetV2 pré-treinada no ImageNet e aplica Transfer
Learning e Fine-Tuning supervisionado nas imagens de dataset/Real/ e dataset/AI/.

O que faz:
  1. Carrega as imagens rotuladas de dataset/Real (classe 0) e dataset/AI (classe 1)
  2. Aplica data augmentation forense (flips, rotações suaves, jitter sutil)
  3. Substitui a cabeça de classificação por uma MLP regularizada
  4. Executa treinamento em 2 fases:
       - Fase 1 (Warmup): Backbone congelado, treina apenas a nova cabeça classificadora
       - Fase 2 (Fine-Tuning): Descongela camadas superiores com taxa de aprendizado reduzida
  5. Salva os pesos treinados em models/spectra_transfer_model.pt e metadados em models/transfer_model_meta.json

Como usar:
  python train_transfer_learning.py
  python train_transfer_learning.py --epochs 15 --backbone efficientnet_v2_s --batch-size 8
"""
import os
import sys
import time
import json
import argparse
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(_HERE, 'dataset', 'Real')
AI_DIR = os.path.join(_HERE, 'dataset', 'AI')
MODELS_DIR = os.path.join(_HERE, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'spectra_transfer_model.pt')
META_PATH = os.path.join(MODELS_DIR, 'transfer_model_meta.json')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}


def list_image_files(directory: str):
    """Lista todos os arquivos de imagem válidos em um diretório."""
    if not os.path.isdir(directory):
        return []
    paths = []
    for name in sorted(os.listdir(directory)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            paths.append(os.path.join(directory, name))
    return paths


def create_model(backbone_name: str = 'efficientnet_v2_s', num_classes: int = 2):
    """Cria a arquitetura base pré-treinada e substitui a cabeça classificadora."""
    import torch
    import torch.nn as nn
    import torchvision.models as models

    print(f"Carregando backbone pré-treinado: {backbone_name} (ImageNet-1K)...")

    if backbone_name == 'efficientnet_v2_s':
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )
    elif backbone_name == 'efficientnet_v2_m':
        weights = models.EfficientNet_V2_M_Weights.DEFAULT
        model = models.efficientnet_v2_m(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )
    elif backbone_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Backbone não suportado: {backbone_name}. Escolha: efficientnet_v2_s, efficientnet_v2_m, efficientnet_b0")

    return model


class ForensicImageDataset:
    """Dataset simples em PyTorch para pares (imagem, label)."""
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, label


def train_model(args):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        import torchvision.transforms as transforms
    except ImportError:
        print("ERRO: PyTorch ou TorchVision não estão instalados.")
        print("Instale com: pip install torch torchvision")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"\nDispositivo de processamento: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Carregar dataset
    real_paths = list_image_files(REAL_DIR)
    ai_paths = list_image_files(AI_DIR)

    n_real = len(real_paths)
    n_ai = len(ai_paths)

    print(f"\nImagens encontradas no dataset:")
    print(f"  - Reais (classe 0): {n_real}")
    print(f"  - IA    (classe 1): {n_ai}")
    print(f"  - Total:           {n_real + n_ai}")

    if n_real == 0 or n_ai == 0:
        print("\nERRO: Ambas as pastas dataset/Real e dataset/AI devem conter ao menos uma imagem.")
        sys.exit(1)

    if n_real < 3 or n_ai < 3:
        print("\nAVISO: Dataset pequeno (<3 amostras por classe).")
        print("Recomendado adicionar ao menos 15-30 imagens por classe para maior generalização.")

    all_paths = real_paths + ai_paths
    all_labels = [0] * n_real + [1] * n_ai

    # 2. Transformações forenses
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ForensicImageDataset(all_paths, all_labels, transform=train_transform)
    eval_dataset = ForensicImageDataset(all_paths, all_labels, transform=eval_transform)

    batch_size = min(args.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # 3. Inicializar Modelo
    model = create_model(args.backbone, num_classes=2)
    model.to(device)

    # Pesos de classe para lidar com possível desbalanceamento
    weight_real = (n_real + n_ai) / (2.0 * n_real)
    weight_ai = (n_real + n_ai) / (2.0 * n_ai)
    class_weights = torch.tensor([weight_real, weight_ai], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- FASE 1: Warmup da Cabeça Classificadora (Backbone congelado) ---
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    if warmup_epochs > 0:
        print(f"\n{'='*55}")
        print(f"FASE 1: Warmup da cabeça classificadora ({warmup_epochs} épocas)")
        print(f"Backbone congelado, treinando apenas a camada MLP...")
        print(f"{'='*55}")

        # Congela features do backbone
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr_head, weight_decay=1e-2)

        for epoch in range(1, warmup_epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(f"  [Warmup {epoch}/{warmup_epochs}] Loss: {epoch_loss:.4f} | Acurácia: {epoch_acc:.1%}")

    # --- FASE 2: Fine-Tuning Parcial (Descongelando camadas superiores) ---
    fine_tune_epochs = args.epochs - warmup_epochs
    if fine_tune_epochs > 0:
        print(f"\n{'='*55}")
        print(f"FASE 2: Fine-Tuning Parcial ({fine_tune_epochs} épocas)")
        print(f"Taxa de aprendizado diferencial: Backbone={args.lr_backbone:.1e}, Cabeça={args.lr_head * 0.5:.1e}")
        print(f"{'='*55}")

        # Descongela as últimas camadas do backbone (stages finais)
        for param in model.features.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW([
            {'params': model.features.parameters(), 'lr': args.lr_backbone},
            {'params': model.classifier.parameters(), 'lr': args.lr_head * 0.5}
        ], weight_decay=1e-2)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-6)

        best_acc = 0.0

        for epoch in range(1, fine_tune_epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            scheduler.step()
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(f"  [Fine-Tune {epoch}/{fine_tune_epochs}] Loss: {epoch_loss:.4f} | Acurácia Treino: {epoch_acc:.1%}")

    # 4. Avaliação Final
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilidade da classe IA
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    final_accuracy = float(np.mean(all_preds == all_targets))

    # Matriz de Confusão
    tp = int(np.sum((all_preds == 1) & (all_targets == 1)))
    fp = int(np.sum((all_preds == 1) & (all_targets == 0)))
    tn = int(np.sum((all_preds == 0) & (all_targets == 0)))
    fn = int(np.sum((all_preds == 0) & (all_targets == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 5. Salvar Modelo e Metadados
    os.makedirs(MODELS_DIR, exist_ok=True)

    checkpoint = {
        'backbone': args.backbone,
        'state_dict': model.state_dict(),
        'num_classes': 2,
        'class_to_idx': {'real': 0, 'ai': 1},
        'input_size': [224, 224],
        'accuracy': final_accuracy,
        'f1_score': f1,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_real': n_real,
        'n_ai': n_ai
    }

    torch.save(checkpoint, MODEL_PATH)

    meta = {
        'backbone': args.backbone,
        'accuracy': round(final_accuracy, 4),
        'f1_score': round(f1, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        'n_real': n_real,
        'n_ai': n_ai,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'epochs': args.epochs,
        'model_file': 'models/spectra_transfer_model.pt'
    }

    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*55}")
    print(f"TRANSFER LEARNING CONCLUÍDO COM SUCESSO!")
    print(f"{'='*55}")
    print(f"Modelo salvo em:    {MODEL_PATH}")
    print(f"Metadados em:       {META_PATH}")
    print(f"Backbone:           {args.backbone}")
    print(f"Acurácia final:     {final_accuracy:.1%} ({np.sum(all_preds == all_targets)}/{len(all_targets)})")
    print(f"F1-Score:           {f1:.3f}")
    print(f"Matriz de confusão: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"{'='*55}")
    print("\nPróximos passos:")
    print("  1. Reinicie ou execute o servidor: python app.py")
    print("  2. O analisador Neural já detectará e utilizará o modelo treinado automaticamente!")
    print("  3. Sempre que adicionar mais imagens a dataset/Real ou dataset/AI, rode novamente:")
    print("     python train_transfer_learning.py")


def main():
    parser = argparse.ArgumentParser(description="Treina Transfer Learning para detecção de IA no Spectra")
    parser.add_argument('--backbone', type=str, default='efficientnet_v2_s',
                        choices=['efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_b0'],
                        help="Arquitetura de backbone pré-treinada (padrão: efficientnet_v2_s)")
    parser.add_argument('--epochs', type=int, default=12,
                        help="Total de épocas de treinamento (padrão: 12)")
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help="Épocas de aquecimento da cabeça com backbone congelado (padrão: 3)")
    parser.add_argument('--batch-size', type=int, default=8,
                        help="Tamanho do lote / batch size (padrão: 8)")
    parser.add_argument('--lr-head', type=float, default=1e-3,
                        help="Taxa de aprendizado da cabeça classificadora (padrão: 0.001)")
    parser.add_argument('--lr-backbone', type=float, default=2e-5,
                        help="Taxa de aprendizado do backbone no fine-tuning (padrão: 0.00002)")
    parser.add_argument('--cpu', action='store_true',
                        help="Forçar uso de CPU mesmo se GPU estiver disponível")

    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()
