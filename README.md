# Spectra

**Analisador Forense de Imagens com IA** — detecta se uma imagem foi gerada por inteligência artificial ou capturada por uma camera fisica, utilizando analise forense multi-espectral.

---

## Visao Geral

O Spectra e uma aplicacao web que combina seis analisadores forenses independentes para produzir uma pontuacao de autenticidade composta para qualquer imagem enviada. Cada analisador examina uma camada de sinal distinta — desde metadados EXIF e artefatos de compressao JPEG ate padroes espectrais de Fourier e distribuicoes estatisticas em nivel de pixel — e os resultados sao agregados com pesos calibrados empiricamente.

O sistema e projetado para ser transparente: cada pontuacao e acompanhada pelas saidas individuais de cada analisador, achados forenses principais e um nivel de confianca derivado da concordancia entre os analisadores.

---

## Arquitetura

```
spectra/
├── app.py                   # Aplicacao Flask, pontuacao composta, endpoints da API
├── main.py                  # Ponto de entrada com banner de inicializacao
├── calibrate.py             # Ferramenta de calibracao empirica (AUC + threshold de Youden)
├── test_accuracy.py         # Suite de avaliacao de acuracia
├── requirements.txt
├── analyzers/
│   ├── metadata_analyzer.py    # EXIF, C2PA, tags de geradores de IA
│   ├── ela_analyzer.py         # Error Level Analysis (residuos de recompressao JPEG)
│   ├── spectral_analyzer.py    # Analise no dominio de frequencia de Fourier
│   ├── noise_analyzer.py       # Modelo de ruido de sensor Poisson-Gaussiano
│   ├── statistical_analyzer.py # Entropia de Shannon, Lei de Benford em gradientes
│   └── artifact_analyzer.py    # Nitidez de bordas, gradiente de profundidade de campo
├── dataset/
│   ├── real/                # Fotografias reais rotuladas (para calibracao)
│   └── ai/                  # Imagens geradas por IA rotuladas (para calibracao)
├── static/                  # Frontend (HTML/CSS/JS)
├── uploads/                 # Diretorio de upload temporario (limpeza automatica)
└── resultados/              # Relatorios de saida da calibracao
```

---

## Analisadores

| Analisador | Sinal examinado | AUC (dataset de calibracao) |
|---|---|---|
| Statistical | Entropia de Shannon por canal, Lei de Benford em gradientes | 0.764 |
| Noise | Correlacao Poisson-Gaussiana, desvio-padrao do canal azul | 0.563 |
| ELA | Distribuicao do nivel de erro de recompressao JPEG | 0.514 |
| Metadata | Campos EXIF de camera, tags de geradores de IA, proveniencia C2PA | Dinamico |
| Spectral | Espectro de potencia de Fourier, ajuste lei de potencia 1/f, planura HF | 0.403 |
| Artifacts | Coeficiente de variacao de nitidez de bordas, gradiente de bokeh | 0.403 |

A pontuacao composta e calculada como uma media ponderada. Os pesos sao ajustados dinamicamente com base na presenca de sinal de metadados: se assinaturas fortes de IA (ex: tags de prompt do Midjourney, C2PA) ou EXIF autentico de camera (Make, Model, ISO, GPS) forem detectados, o peso do analisador de metadados e aumentado significativamente. Se o EXIF foi removido (comum em redes sociais), o analisador de metadados e excluido inteiramente do composto.

Analisadores que falham em tempo de execucao sao excluidos do composto em vez de substituidos por uma pontuacao neutra.

---

## Interpretacao da Pontuacao

| Faixa | Veredicto |
|---|---|
| 0 -- 25 | Provavelmente Autentica (Foto Real) |
| 26 -- 40 | Tendencia Autentica (Baixo Risco de IA) |
| 41 -- 59 | Inconclusivo (Sinais Mistos) |
| 60 -- 74 | Suspeito — Tendencia de IA |
| 75 -- 100 | Provavelmente Gerada por IA |

O nivel de confianca (`high`, `medium`, `low`) e calculado separadamente a partir da razao de concordancia entre analisadores e do desvio padrao das pontuacoes dos seis analisadores.

---

## Requisitos

- Python 3.10 ou superior
- Dependencias listadas em `requirements.txt`:

```
Flask==3.1.1
Pillow==11.2.1
numpy==2.2.6
scipy==1.15.3
exifread==3.1.0
```

---

## Instalacao

```bash
# Clone o repositorio
git clone <url-do-repositorio>
cd spectra

# Crie e ative um ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Instale as dependencias
pip install -r requirements.txt
```

---

## Executando o Servidor

```bash
python main.py
```

O servidor inicia em `http://localhost:5000`. Abra esse endereco no navegador para acessar a interface web.

Alternativamente, para desenvolvimento com o reloader do Flask:

```bash
python app.py
```

---

## API

### `POST /analyze`

Aceita um upload multipart e retorna a analise forense completa.

**Requisicao**

```
Content-Type: multipart/form-data
Campo: image (arquivo)
```

Formatos suportados: `jpg`, `jpeg`, `png`, `webp`, `bmp`, `tiff`. Tamanho maximo: 32 MB.

**Resposta (sucesso)**

```json
{
  "success": true,
  "score": 72,
  "verdict": "Suspeito — Tendencia de IA",
  "verdict_level": "suspect",
  "confidence": "medium",
  "agreement": 0.67,
  "summary": {
    "text": "...",
    "key_findings": ["..."]
  },
  "analyses": {
    "metadata": { "score": 85, "details": { "metrics": {} }, "findings": [] },
    "ela":      { "score": 55, "details": { "metrics": {} }, "findings": [] },
    "...": {}
  },
  "failed_analyzers": [],
  "image_info": {
    "filename": "foto.jpg",
    "format": "JPEG",
    "mode": "RGB",
    "size": [1024, 768],
    "file_size": 204800
  },
  "processing_time": 1.34
}
```

### `GET /api/health`

Retorna o status do servidor e a lista de analisadores ativos.

```json
{
  "status": "ok",
  "version": "2.0.0-calibrated",
  "analyzers": ["metadata", "ela", "spectral", "noise", "statistical", "artifacts"]
}
```

---

## Calibracao

Os thresholds das metricas individuais sao calibrados empiricamente com `calibrate.py`. O script executa todos os analisadores sobre um dataset rotulado e calcula, para cada metrica interna:

- **AUC** (estatistica U de Mann-Whitney / Wilcoxon baseada em ranking, sem dependencia do sklearn)
- **Threshold otimo** via estatistica J de Youden (maximiza TPR - FPR)

O script tambem avalia a pontuacao composta atual contra uma matriz de confusao e treina uma regressao logistica simples nos seis scores dos analisadores para sugerir ajustes de pesos baseados em dados.

### Como Usar

Organize seu dataset:

```
dataset/
  real/    # fotografias reais — fontes variadas, com e sem EXIF
  ai/      # imagens de IA — multiplas ferramentas (Midjourney, SDXL, Flux, DALL-E, ...)
```

Execute a calibracao:

```bash
python calibrate.py \
    --real-dir dataset/real \
    --ai-dir dataset/ai \
    --project-root . \
    --output-dir resultados/
```

**Arquivos de saida:**

| Arquivo | Conteudo |
|---|---|
| `calibration_report.md` | Relatorio legivel: ranking de metricas por AUC, matrizes de confusao, pesos logisticos |
| `calibration_data.csv` | Todas as metricas extraidas por imagem — adequado para analise externa ou plotagem |
| `calibration_raw.json` | Mesmos dados em formato JSON para uso programatico |

Um minimo de 30 imagens por classe e recomendado para estimativas de AUC estaveis. 150 ou mais por classe e o ideal para calibracao de producao.

---

## Recomendacoes para o Dataset

Para que a calibracao seja significativa:

- **Imagens reais**: inclua fotos tiradas diretamente de cameras e fotos baixadas do WhatsApp, Instagram ou Twitter (EXIF removido), para testar robustez em diferentes condicoes de metadados.
- **Imagens de IA**: inclua multiplos geradores (Midjourney, Stable Diffusion XL, Flux, DALL-E) e varie a presenca ou ausencia de metadados.
- **Nao reutilize o dataset de calibracao para validacao de acuracia.** Apos ajustar os thresholds, valide em um conjunto separado para distinguir melhoria genuina de overfitting no dataset de calibracao.

---

## Licenca

Este projeto nao possui uma licenca definida. Todos os direitos reservados, salvo indicacao contraria.
