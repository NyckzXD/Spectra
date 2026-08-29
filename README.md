# Spectra

**Analisador Forense de Imagens com IA & Transfer Learning** — detecta se uma imagem foi gerada por inteligência artificial ou capturada por uma câmera física, combinando visão computacional profunda (Transfer Learning com EfficientNetV2) e análise forense multi-espectral.

---

## Visão Geral

O Spectra é uma aplicação web que combina múltiplos analisadores forenses e neurais independentes para produzir uma pontuação de autenticidade composta para qualquer imagem enviada. Cada analisador examina uma camada de sinal distinta — desde redes neurais convolucionais profundas e embeddings CLIP até metadados EXIF, decomposição Wavelet e estatística em nível de pixel — com agregação calibrada.

---

## Arquitetura

```
spectra/
├── app.py                         # Aplicação Flask, pontuação composta, endpoints da API
├── main.py                        # Ponto de entrada com banner de inicialização
├── train_transfer_learning.py     # Pipeline de Transfer Learning (EfficientNetV2)
├── build_clip_prototypes.py       # Gerador de protótipos de embeddings CLIP
├── calibrate.py                   # Ferramenta de calibração estatística (AUC + Youden)
├── test_accuracy.py               # Suíte de avaliação de acurácia
├── requirements.txt
├── analyzers/
│   ├── neural_analyzer.py         # Deep Learning via Transfer Learning (EfficientNetV2)
│   ├── clip_analyzer.py           # Espaço latente multimodal (CLIP ViT-B/32)
│   ├── wavelet_analyzer.py        # Decomposição multi-escala DWT 2D
│   ├── metadata_analyzer.py       # EXIF, C2PA, tags de geradores de IA
│   ├── noise_analyzer.py          # Modelo de ruído de sensor Poisson-Gaussiano
│   ├── spectral_analyzer.py       # Análise no domínio de frequência de Fourier (FFT)
│   ├── statistical_analyzer.py    # Entropia de Shannon, Lei de Benford em gradientes
│   └── artifact_analyzer.py       # Nitidez de bordas, gradiente de profundidade de campo
├── models/
│   ├── spectra_transfer_model.pt  # Pesos do modelo EfficientNetV2 fine-tuned
│   ├── transfer_model_meta.json   # Metadados e métricas do treinamento
│   └── clip_prototypes.npz        # Centroides dos embeddings CLIP
├── dataset/
│   ├── Real/                      # Fotografias reais rotuladas
│   └── AI/                        # Imagens geradas por IA rotuladas
├── static/                        # Frontend (HTML/CSS/JS)
└── resultados/                    # Relatórios de saída da calibração
```

---

## Analisadores

| Analisador | Sinal examinado | Tipo de Análise |
|---|---|---|
| **Neural (Transfer Learning)** | Padrões profundos aprendidos por fine-tuning no EfficientNetV2 | Deep Learning CNN |
| **CLIP** | Proximidade em espaço latente multimodal ViT-B/32 | Vision Transformer |
| **Wavelet (DWT)** | Kurtose e energia de coeficientes em múltiplas escalas | Frequência/Espacial |
| **Statistical** | Entropia de Shannon por canal, Lei de Benford em gradientes | Estatística |
| **Noise** | Correlação Poisson-Gaussiana, desvio-padrão do canal azul | Ruído de Sensor |
| **Metadata** | Campos EXIF de câmera, tags de geradores de IA, proveniência C2PA | Metadados |
| **Spectral** | Espectro de potência de Fourier, ajuste lei de potência 1/f, planura HF | Frequência (FFT) |
| **Artifacts** | Coeficiente de variação de nitidez de bordas, gradiente de bokeh | Óptica e Bordas |

---

## Transfer Learning (EfficientNetV2)

Em vez de treinar redes neurais do zero (o que exigiria milhões de imagens e alto custo computacional), o Spectra utiliza **Transfer Learning**:
1. **Backbone Pré-treinado:** Utiliza o `EfficientNetV2-S` pré-treinado no ImageNet-1K.
2. **Nova Cabeça Classificadora:** MLP customizada com Dropout, GELU e BatchNorm.
3. **Treinamento em Duas Fases:**
   - *Fase 1 (Warmup):* Backbone congelado para estabilizar os pesos da nova cabeça.
   - *Fase 2 (Fine-Tuning):* Descongelamento parcial com taxa de aprendizado diferenciada (`lr_backbone = 2e-5`, `lr_head = 5e-4`) e agendador `CosineAnnealingLR`.

### Como Treinar / Retreinar

Após adicionar imagens em `dataset/Real/` e `dataset/AI/`:

```bash
python train_transfer_learning.py
```

Opções adicionais:
```bash
python train_transfer_learning.py --epochs 15 --batch-size 8 --backbone efficientnet_v2_s
```

---

## Interpretação da Pontuação

| Faixa | Veredicto |
|---|---|
| 0 -- 25 | Provavelmente Autêntica (Foto Real) |
| 26 -- 40 | Tendência Autêntica (Baixo Risco de IA) |
| 41 -- 59 | Inconclusivo (Sinais Mistos) |
| 60 -- 74 | Suspeito — Tendência de IA |
| 75 -- 100 | Provavelmente Gerada por IA |

---

## Executando o Servidor

```bash
python app.py
```

Acesse no navegador: `http://localhost:5000`
