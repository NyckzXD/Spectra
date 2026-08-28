document.addEventListener('DOMContentLoaded', () => {
    // --- DOM References ---
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const retryBtn = document.getElementById('retryBtn');
    const resetBtn = document.getElementById('resetBtn');

    const uploadSection = document.getElementById('uploadSection');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');

    const uploadContent = document.getElementById('uploadContent');
    const uploadPreview = document.getElementById('uploadPreview');
    const uploadPreviewImg = document.getElementById('uploadPreviewImg');
    const previewFileName = document.getElementById('previewFileName');
    const previewFileSize = document.getElementById('previewFileSize');

    const imagePreview = document.getElementById('imagePreview');
    const imageInfo = document.getElementById('imageInfo');
    const verdictBanner = document.getElementById('verdictBanner');
    const verdictSummary = document.getElementById('verdictSummary');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const confidenceText = document.getElementById('confidenceText');
    const keyFindingsList = document.getElementById('keyFindingsList');
    const keyFindingsSection = document.getElementById('keyFindingsSection');
    const analysisGrid = document.getElementById('analysisGrid');
    const processingTime = document.getElementById('processingTime');
    const errorMessage = document.getElementById('errorMessage');

    let selectedFile = null;

    // --- Utility Functions ---
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
    }

    function showSection(section) {
        [uploadSection, loadingSection, resultsSection, errorSection].forEach(s => {
            s.classList.add('hidden');
        });
        section.classList.remove('hidden');
    }

    function getColorForScore(score) {
        if (score <= 25) return '#10B981';
        if (score <= 50) return '#F59E0B';
        if (score <= 75) return '#F97316';
        return '#EF4444';
    }

    // --- File Selection & Preview ---
    function showPreview(file) {
        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreviewImg.src = e.target.result;
            imagePreview.src = e.target.result;
        };
        reader.readAsDataURL(file);

        previewFileName.textContent = file.name;
        previewFileSize.textContent = formatFileSize(file.size);

        // Switch from upload content to preview
        uploadContent.classList.add('hidden');
        uploadPreview.classList.remove('hidden');
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        uploadContent.classList.remove('hidden');
        uploadPreview.classList.add('hidden');
        uploadPreviewImg.src = '';
    }

    function handleFile(file) {
        if (!file || !file.type.match('image.*')) {
            showError('Selecione um arquivo de imagem válido (JPEG, PNG, WebP, BMP, TIFF).');
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            showError('O arquivo excede o limite de 16 MB.');
            return;
        }

        showPreview(file);
    }

    // --- Event Listeners ---
    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('.primary-btn') || e.target.closest('.secondary-btn')) return;
        if (!uploadPreview.classList.contains('hidden')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Clipboard paste (Ctrl+V)
    document.addEventListener('paste', (e) => {
        // Only handle if upload section is visible
        if (uploadSection.classList.contains('hidden')) return;

        const items = e.clipboardData?.items;
        if (!items) return;

        for (const item of items) {
            if (item.type.startsWith('image/')) {
                const file = item.getAsFile();
                if (file) {
                    handleFile(file);
                    e.preventDefault();
                    break;
                }
            }
        }
    });

    // Analyze button
    analyzeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (selectedFile) {
            analyzeImage(selectedFile);
        }
    });

    // Cancel preview
    cancelBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // Retry from error
    retryBtn.addEventListener('click', () => {
        showSection(uploadSection);
        resetUpload();
    });

    // Reset from results
    resetBtn.addEventListener('click', () => {
        showSection(uploadSection);
        resetUpload();
    });

    // --- Error Display ---
    function showError(message) {
        errorMessage.textContent = message;
        showSection(errorSection);
    }

    // --- Analysis ---
    async function analyzeImage(file) {
        showSection(loadingSection);

        // Animate loading steps
        const steps = document.querySelectorAll('.progress-steps .step');
        let currentStep = 0;

        steps.forEach(s => {
            s.classList.remove('active', 'done');
            s.querySelector('.step-icon').textContent = '◯';
        });
        steps[0].classList.add('active');

        const stepInterval = setInterval(() => {
            if (currentStep < steps.length) {
                steps[currentStep].classList.remove('active');
                steps[currentStep].classList.add('done');
                steps[currentStep].querySelector('.step-icon').textContent = '';
            }
            currentStep++;
            if (currentStep < steps.length) {
                steps[currentStep].classList.add('active');
            } else {
                // Loop back for long analyses
                currentStep = 0;
                steps.forEach(s => {
                    s.classList.remove('active', 'done');
                    s.querySelector('.step-icon').textContent = '◯';
                });
                steps[0].classList.add('active');
            }
        }, 900);

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            clearInterval(stepInterval);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Erro HTTP ${response.status}`);
            }

            const data = await response.json();

            if (data.success === false) {
                showError(data.error || 'A análise falhou por um erro desconhecido.');
                return;
            }

            renderResults(data);

        } catch (error) {
            clearInterval(stepInterval);
            console.error('Analysis failed:', error);
            showError(error.message || 'Falha ao conectar com o servidor de análise. Verifique sua conexão.');
        }
    }

    // --- Results Rendering ---
    function renderResults(data) {
        showSection(resultsSection);

        renderImageInfo(data.image_info, data.processing_time);
        drawScoreGauge(data.score);
        renderVerdict(data.score, data.verdict, data.confidence, data.summary);
        renderKeyFindings(data.summary?.key_findings);
        renderAnalysisCards(data.analyses);

        if (data.processing_time) {
            processingTime.textContent = `Processado em ${data.processing_time}s`;
        }

        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
    }

    function renderImageInfo(info, time) {
        if (!info || !imageInfo) return;

        const sizeStr = Array.isArray(info.size) ? info.size.join(' × ') : info.size;
        const fileSizeStr = formatFileSize(info.file_size);

        imageInfo.innerHTML = `
            <div class="info-item"><span class="info-label">Arquivo</span><span class="info-value">${info.filename}</span></div>
            <div class="info-item"><span class="info-label">Formato</span><span class="info-value">${info.format}</span></div>
            <div class="info-item"><span class="info-label">Modo</span><span class="info-value">${info.mode || '—'}</span></div>
            <div class="info-item"><span class="info-label">Dimensões</span><span class="info-value">${sizeStr} px</span></div>
            <div class="info-item"><span class="info-label">Tamanho</span><span class="info-value">${fileSizeStr}</span></div>
        `;
    }

    function renderVerdict(score, verdictText, confidence, summary) {
        const color = getColorForScore(score);

        verdictBanner.textContent = verdictText;
        verdictBanner.style.color = color;

        // Confidence badge
        confidenceBadge.className = `confidence-badge ${confidence || 'medium'}`;
        const confLabels = { high: 'Alta', medium: 'Média', low: 'Baixa' };
        confidenceText.textContent = `Confiança: ${confLabels[confidence] || 'Média'}`;

        // Summary text
        if (summary?.text) {
            verdictSummary.textContent = summary.text;
            verdictSummary.style.display = '';
        } else {
            verdictSummary.style.display = 'none';
        }

        // Hero border accent
        const verdictHero = document.getElementById('verdictHero');
        verdictHero.style.borderColor = `${color}30`;
    }

    function renderKeyFindings(findings) {
        if (!findings || findings.length === 0) {
            keyFindingsSection.classList.add('hidden');
            return;
        }

        keyFindingsSection.classList.remove('hidden');
        keyFindingsList.innerHTML = '';

        const icons = ['', '', '', '', ''];
        findings.forEach((finding, i) => {
            const chip = document.createElement('div');
            chip.className = 'finding-chip';
            chip.innerHTML = `<span class="finding-icon">${icons[i % icons.length]}</span><span>${finding}</span>`;
            keyFindingsList.appendChild(chip);
        });
    }

    function drawScoreGauge(finalScore) {
        const canvas = document.getElementById('scoreGauge');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;

        canvas.width = 220 * dpr;
        canvas.height = 220 * dpr;
        canvas.style.width = '220px';
        canvas.style.height = '220px';
        ctx.scale(dpr, dpr);

        const centerX = 110;
        const centerY = 110;
        const radius = 88;
        const lineWidth = 14;

        let currentScore = 0;
        const animationDuration = 2200;
        const startTime = performance.now();

        function animate(time) {
            const elapsed = time - startTime;
            const progress = Math.min(elapsed / animationDuration, 1);
            const easeOut = 1 - Math.pow(1 - progress, 3);
            currentScore = Math.round(easeOut * finalScore);

            ctx.clearRect(0, 0, 220, 220);

            // Background track
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, 2.25 * Math.PI);
            ctx.lineWidth = lineWidth;
            ctx.strokeStyle = '#E2E8F0';
            ctx.lineCap = 'round';
            ctx.stroke();

            // Progress arc
            const startAngle = 0.75 * Math.PI;
            const endAngle = startAngle + (currentScore / 100) * (1.5 * Math.PI);
            const color = getColorForScore(currentScore);

            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, startAngle, endAngle);
            ctx.lineWidth = lineWidth;
            ctx.strokeStyle = color;
            ctx.lineCap = 'round';
            ctx.stroke();

            // Score text
            ctx.fillStyle = '#0F172A';
            ctx.font = `bold 46px Outfit, Inter, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`${currentScore}`, centerX, centerY - 6);

            // Label
            ctx.font = `600 10px Inter, sans-serif`;
            ctx.fillStyle = color;
            ctx.fillText('PROBABILIDADE AI', centerX, centerY + 26);

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        }

        requestAnimationFrame(animate);
    }

    // --- Analysis Cards ---
    function formatValue(value) {
        if (typeof value === 'boolean') return value ? 'Sim' : 'Não';
        if (typeof value === 'number') return Number.isInteger(value) ? value : value.toFixed(3);
        if (Array.isArray(value)) return value.map(v => typeof v === 'number' ? v.toFixed(2) : v).join(', ');
        if (typeof value === 'object' && value !== null) {
            return Object.entries(value).map(([k, v]) => `${k}: ${formatValue(v)}`).join(' | ');
        }
        return String(value);
    }

    function renderDetailsTable(obj) {
        if (!obj || Object.keys(obj).length === 0) return '';
        let html = '<table class="metadata-table"><tbody>';
        for (const [k, v] of Object.entries(obj)) {
            const val = typeof v === 'string' && v.length > 100 ? v.substring(0, 100) + '…' : formatValue(v);
            html += `<tr><td><strong>${k}</strong></td><td>${val}</td></tr>`;
        }
        html += '</tbody></table>';
        return html;
    }

    function renderDetailsList(obj) {
        if (!obj || Object.keys(obj).length === 0) return '';
        let html = '<ul class="details-list">';
        for (const [k, v] of Object.entries(obj)) {
            const formattedKey = k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            html += `<li><strong>${formattedKey}:</strong> ${formatValue(v)}</li>`;
        }
        html += '</ul>';
        return html;
    }

    function renderAnalysisCards(analyses) {
        analysisGrid.innerHTML = '';

        const cardDefs = [
            { key: 'metadata', icon: '', title: 'Análise de Metadados', desc: 'EXIF, C2PA, assinaturas de IA' },
            { key: 'ela', icon: '', title: 'Error Level Analysis', desc: 'Compressão JPEG e uniformidade' },
            { key: 'spectral', icon: '', title: 'Análise Espectral (FFT)', desc: 'Domínio de frequência e picos' },
            { key: 'noise', icon: '', title: 'Análise de Ruído', desc: 'Padrões de ruído e PRNU' },
            { key: 'statistical', icon: '', title: 'Análise Estatística', desc: 'Benford, GLCM, entropia' },
            { key: 'artifacts', icon: '', title: 'Análise de Artefatos', desc: 'JPEG grid, checkerboard, edges' }
        ];

        cardDefs.forEach((def, index) => {
            const data = analyses[def.key];
            if (!data) return;

            const card = document.createElement('div');
            card.className = 'analysis-card';
            card.style.animationDelay = `${index * 0.08}s`;

            const color = getColorForScore(data.score);

            let contentHTML = '';

            // Findings
            if (data.findings && data.findings.length > 0) {
                contentHTML += '<ul class="card-findings">';
                data.findings.forEach(f => {
                    contentHTML += `<li>${f}</li>`;
                });
                contentHTML += '</ul>';
            }

            // Details
            if (data.details) {
                if (def.key === 'metadata' && data.details.metadata) {
                    const metaEntries = Object.entries(data.details.metadata);
                    if (metaEntries.length > 0) {
                        // Show max 8 entries
                        const limited = Object.fromEntries(metaEntries.slice(0, 8));
                        contentHTML += renderDetailsTable(limited);
                        if (metaEntries.length > 8) {
                            contentHTML += `<p style="color: var(--text-muted); font-size: 11px; margin-top: 6px;">+${metaEntries.length - 8} campos adicionais</p>`;
                        }
                    }
                } else if (data.details.metrics) {
                    contentHTML += renderDetailsList(data.details.metrics);
                }
            }

            // Visualization
            if (data.visualization) {
                contentHTML += `<img src="data:image/png;base64,${data.visualization}" class="card-visualization" alt="${def.title}" title="Clique para ampliar">`;
            } else if (def.key === 'statistical' && data.histogram_data) {
                contentHTML += `<div style="height: 120px; width: 100%; margin-top: 12px;"><canvas id="chart-${def.key}" class="card-visualization"></canvas></div>`;
            }

            card.innerHTML = `
                <div class="card-header">
                    <span class="card-icon">${def.icon}</span>
                    <div>
                        <h3 class="card-title">${def.title}</h3>
                    </div>
                </div>
                <div class="score-bar-container">
                    <div class="score-bar-fill" style="background-color: ${color};" data-target="${data.score}%"></div>
                </div>
                <div class="card-content">
                    <span class="card-score-text" style="color: ${color}">Score: ${data.score}%</span>
                    ${contentHTML}
                </div>
            `;

            analysisGrid.appendChild(card);

            // Animate score bar
            setTimeout(() => {
                const bar = card.querySelector('.score-bar-fill');
                if (bar) bar.style.width = bar.getAttribute('data-target');
            }, 300 + (index * 80));

            // Render histogram chart
            if (def.key === 'statistical' && data.histogram_data) {
                setTimeout(() => renderHistogram(`chart-${def.key}`, data.histogram_data), 600);
            }
        });

        // Lightbox for visualization images
        setupLightbox();
    }

    function renderHistogram(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || typeof Chart === 'undefined') return;

        new Chart(canvas, {
            type: 'bar',
            data: {
                labels: Array.from({ length: 32 }, (_, i) => i * 8),
                datasets: [
                    { label: 'R', data: data.r || [], backgroundColor: 'rgba(239, 68, 68, 0.7)', barPercentage: 1.0, categoryPercentage: 1.0 },
                    { label: 'G', data: data.g || [], backgroundColor: 'rgba(16, 185, 129, 0.7)', barPercentage: 1.0, categoryPercentage: 1.0 },
                    { label: 'B', data: data.b || [], backgroundColor: 'rgba(89, 165, 216, 0.7)', barPercentage: 1.0, categoryPercentage: 1.0 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: false, stacked: true },
                    y: { display: false, stacked: true }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: false }
                },
                interaction: { mode: 'index', intersect: false }
            }
        });
    }

    // --- Lightbox ---
    function setupLightbox() {
        document.querySelectorAll('.card-visualization').forEach(img => {
            if (img.tagName !== 'IMG') return;
            img.addEventListener('click', () => {
                const overlay = document.createElement('div');
                overlay.className = 'lightbox-overlay';
                overlay.innerHTML = `<img src="${img.src}" alt="Visualização ampliada">`;
                overlay.addEventListener('click', () => overlay.remove());
                document.addEventListener('keydown', function handler(e) {
                    if (e.key === 'Escape') {
                        overlay.remove();
                        document.removeEventListener('keydown', handler);
                    }
                });
                document.body.appendChild(overlay);
            });
        });
    }
});
