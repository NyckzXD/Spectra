# Spectra

**Analisador Forense de Imagens com IA** — detecta se uma imagem foi gerada por inteligência artificial ou capturada por uma câmera física, combinando visão computacional profunda com modelo especialista pré-treinado (**Vision Transformer via Hugging Face**) e análise forense multi-espectral.

---

## Visão Geral

O Spectra é uma aplicação web que combina múltiplos analisadores forenses e neurais independentes para produzir uma pontuação de autenticidade composta para qualquer imagem enviada.

### 🌟 Detecção Neural 100% Pronta (Zero Treinamento Local)
O Spectra agora integra diretamente modelos de ponta da comunidade Hugging Face (como `umm-maybe/AI-image-detector`), pré-treinados em **centenas de milhares de imagens sintéticas** (Midjourney, Stable Diffusion, DALL-E, SDXL, Flux) e fotos reais.

**Você não precisa coletar datasets nem treinar modelos localmente**: ao iniciar o servidor e enviar uma imagem, os pesos são baixados automaticamente e a inferência é executada de forma instantânea.

---

## Arquitetura

```
spectra/
├── app.py                         # Aplicação Flask, pontuação composta, endpoints da API
├── main.py                        # Ponto de entrada com banner de inicialização
├── requirements.txt               # Dependências do projeto
├── analyzers/
│   ├── neural_analyzer.py         # Detector Neural Especialista (Hugging Face ViT)
│   ├── clip_analyzer.py           # Espaço latente multimodal (CLIP ViT-B/32)
│   ├── wavelet_analyzer.py        # Decomposição multi-escala DWT 2D
│   ├── metadata_analyzer.py       # EXIF, C2PA, tags de geradores de IA
│   ├── noise_analyzer.py          # Modelo de ruído de sensor Poisson-Gaussiano
│   ├── spectral_analyzer.py       # Análise no domínio de frequência de Fourier (FFT)
│   ├── statistical_analyzer.py    # Entropia de Shannon, Lei de Benford em gradientes
│   └── artifact_analyzer.py       # Nitidez de bordas, gradiente de profundidade de campo
├── static/                        # Frontend (HTML/CSS/JS)
└── dataset/                       # Amostras de referência
```

---

## Analisadores Forenses

| Analisador | Sinal examinado | Tipo de Análise |
|---|---|---|
| **Detector Neural (Hugging Face)** | Vision Transformer pré-treinado em centenas de milhares de imagens de IA | Deep Learning ViT |
| **CLIP** | Proximidade em espaço latente multimodal ViT-B/32 | Vision Transformer |
| **Wavelet (DWT)** | Kurtose e energia de coeficientes em múltiplas escalas | Frequência/Espacial |
| **Statistical** | Entropia de Shannon por canal, Lei de Benford em gradientes | Estatística |
| **Noise** | Correlação Poisson-Gaussiana, desvio-padrão do canal azul | Ruído de Sensor |
| **Metadata** | Campos EXIF de câmera, tags de geradores de IA, proveniência C2PA | Metadados |
| **Spectral** | Espectro de potência de Fourier, ajuste lei de potência 1/f, planura HF | Frequência (FFT) |
| **Artifacts** | Coeficiente de variação de nitidez de bordas, gradiente de bokeh | Óptica e Bordas |

---

## Como Instalar e Rodar

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```
*(ou execute o script `install_deps.bat` no Windows)*

### 2. Iniciar o Servidor
```bash
python app.py
```

Acesse no navegador: `http://localhost:5000`
Faça o upload ou cole (`Ctrl+V`) qualquer imagem para obter a análise completa!
