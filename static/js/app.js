document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    
    const uploadSection = document.getElementById('uploadSection');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    
    const imagePreview = document.getElementById('imagePreview');
    const imageInfo = document.getElementById('imageInfo');
    const verdictBanner = document.getElementById('verdictBanner');
    const analysisGrid = document.getElementById('analysisGrid');
    
    const resetBtn = document.getElementById('resetBtn');
    
    // UI Events
    browseBtn.addEventListener('click', () => fileInput.click());
    
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            resultsSection.classList.add('hidden');
            if (errorSection) errorSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
            fileInput.value = '';
        });
    }
    
    // Retry button inside error section
    let retryBtn;
    if (errorSection) {
        retryBtn = document.createElement('button');
        retryBtn.textContent = 'Retry';
        retryBtn.className = 'primary-btn';
        retryBtn.style.marginTop = '1rem';
        retryBtn.addEventListener('click', () => {
            errorSection.classList.add('hidden');
            uploadSection.classList.remove('hidden');
            fileInput.value = '';
        });
        errorSection.querySelector('.error-content').appendChild(retryBtn);
    }

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            showError('Please select a valid image file (JPEG, PNG, WebP).');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
        };
        reader.readAsDataURL(file);

        analyzeImage(file);
    }

    function showError(message) {
        uploadSection.classList.add('hidden');
        loadingSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        
        if (errorSection) {
            errorSection.classList.remove('hidden');
            let msgEl = errorSection.querySelector('.error-message');
            if (!msgEl) {
                msgEl = document.createElement('p');
                msgEl.className = 'error-message';
                errorSection.querySelector('.error-content').insertBefore(msgEl, retryBtn);
            }
            msgEl.textContent = message;
        } else {
            alert("Error: " + message);
        }
    }

    async function analyzeImage(file) {
        uploadSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        if (errorSection) errorSection.classList.add('hidden');

        // Animate loading steps purely for visual effect while waiting
        const steps = document.querySelectorAll('.progress-steps .step');
        let currentStep = 0;
        const stepInterval = setInterval(() => {
            steps.forEach(s => s.classList.remove('active'));
            if (currentStep < steps.length) {
                steps[currentStep].classList.add('active');
                currentStep++;
            } else {
                currentStep = 0;
            }
        }, 800);

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            clearInterval(stepInterval);

            if (data.success === false) {
                showError(data.error || 'Analysis failed due to unknown server error.');
                return;
            }
            
            renderResults(data);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            clearInterval(stepInterval);
            showError('Failed to connect to the analysis server. Please check your connection and try again.');
        }
    }

    function formatValue(value) {
        if (typeof value === 'number') {
            return Number.isInteger(value) ? value : value.toFixed(2);
        }
        if (Array.isArray(value)) {
            return value.map(v => typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : v).join(', ');
        }
        if (typeof value === 'object' && value !== null) {
            return Object.entries(value).map(([k, v]) => `${k}: ${formatValue(v)}`).join(' | ');
        }
        return value;
    }

    function renderDetailsRecursive(obj, isTable = false) {
        if (!obj || Object.keys(obj).length === 0) return '';
        
        let html = '';
        if (isTable) {
            html += '<table class="metadata-table"><tbody>';
            for (const [k, v] of Object.entries(obj)) {
                html += `<tr><td><strong>${k}</strong></td><td>${formatValue(v)}</td></tr>`;
            }
            html += '</tbody></table>';
        } else {
            html += '<ul class="details-list">';
            for (const [k, v] of Object.entries(obj)) {
                const formattedKey = k.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                html += `<li><strong>${formattedKey}:</strong> ${formatValue(v)}</li>`;
            }
            html += '</ul>';
        }
        return html;
    }

    function renderResults(data) {
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        if (resetBtn) resetBtn.classList.remove('hidden');

        // Render Image Info
        if (data.image_info && imageInfo) {
            const sizeStr = Array.isArray(data.image_info.size) ? data.image_info.size.join('x') : data.image_info.size;
            const sizeKB = (data.image_info.file_size / 1024).toFixed(1) + ' KB';
            imageInfo.innerHTML = `
                <div class="info-item"><span>Filename:</span> ${data.image_info.filename}</div>
                <div class="info-item"><span>Format:</span> ${data.image_info.format}</div>
                <div class="info-item"><span>Dimensions:</span> ${sizeStr}</div>
                <div class="info-item"><span>Size:</span> ${sizeKB}</div>
            `;
        }

        drawScoreGauge(data.score);
        renderVerdict(data.score, data.verdict);
        renderAnalysisCards(data.analyses);
        
        // Smooth scroll to results after a brief delay
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    function getColorForScore(score) {
        if (score <= 25) return '#10b981'; // Green (Authentic)
        if (score <= 50) return '#fbbf24'; // Yellow (Inconclusive)
        if (score <= 75) return '#f97316'; // Orange (Suspect)
        return '#ef4444'; // Red (Likely AI)
    }

    function drawScoreGauge(finalScore) {
        const canvas = document.getElementById('scoreGauge');
        const ctx = canvas.getContext('2d');
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = 80;
        const lineWidth = 16;

        let currentScore = 0;
        const animationDuration = 2000;
        const startTime = performance.now();

        function animate(time) {
            const elapsed = time - startTime;
            const progress = Math.min(elapsed / animationDuration, 1);
            
            // Ease out cubic
            const easeOut = 1 - Math.pow(1 - progress, 3);
            currentScore = Math.round(easeOut * finalScore);

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw background track
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, 2.25 * Math.PI);
            ctx.lineWidth = lineWidth;
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            ctx.lineCap = 'round';
            ctx.stroke();

            // Draw progress arc
            const startAngle = 0.75 * Math.PI;
            const endAngle = startAngle + (currentScore / 100) * (1.5 * Math.PI);
            
            const color = getColorForScore(currentScore);

            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, startAngle, endAngle);
            ctx.lineWidth = lineWidth;
            ctx.strokeStyle = color;
            ctx.stroke();

            // Draw text
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 42px Inter';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`${currentScore}`, centerX, centerY - 5);
            
            ctx.font = '500 11px JetBrains Mono';
            ctx.fillStyle = color;
            ctx.fillText('AI PROBABILITY', centerX, centerY + 25);

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        }
        requestAnimationFrame(animate);
    }

    function renderVerdict(score, verdictText) {
        const color = getColorForScore(score);
        verdictBanner.textContent = verdictText;
        verdictBanner.style.color = color;
        verdictBanner.style.borderColor = `${color}40`;
        verdictBanner.style.boxShadow = `0 4px 20px ${color}20`;
    }

    function renderAnalysisCards(analyses) {
        analysisGrid.innerHTML = '';
        
        const cardDefinitions = [
            { key: 'metadata', icon: '🏷️', title: 'Metadata Analysis' },
            { key: 'ela', icon: '🔍', title: 'Error Level Analysis' },
            { key: 'spectral', icon: '📡', title: 'Spectral Analysis' },
            { key: 'noise', icon: '📊', title: 'Noise Analysis' },
            { key: 'statistical', icon: '📈', title: 'Statistical Analysis' },
            { key: 'artifacts', icon: '🔬', title: 'Artifact Analysis' }
        ];

        cardDefinitions.forEach((def, index) => {
            const data = analyses[def.key];
            if (!data) return;

            const card = document.createElement('div');
            card.className = 'analysis-card';
            card.style.animationDelay = `${index * 0.1}s`;

            const color = getColorForScore(data.score);

            let contentHTML = '';

            if (data.findings && data.findings.length) {
                contentHTML += '<ul class="card-findings">';
                data.findings.forEach(f => {
                    contentHTML += `<li>${f}</li>`;
                });
                contentHTML += '</ul>';
            } 
            
            if (data.details) {
                if (def.key === 'metadata' && data.details.metadata) {
                    contentHTML += renderDetailsRecursive(data.details.metadata, true);
                } else if (data.details.metrics) {
                    contentHTML += renderDetailsRecursive(data.details.metrics, false);
                } else {
                    contentHTML += renderDetailsRecursive(data.details, false);
                }
            }

            if (data.visualization) {
                contentHTML += `<img src="data:image/png;base64,${data.visualization}" class="card-visualization" alt="${def.title} visualization">`;
            } else if (def.key === 'statistical' && data.histogram_data) {
                contentHTML += `<div style="height: 120px; width: 100%; margin-top: 1rem;"><canvas id="chart-${def.key}" class="card-visualization"></canvas></div>`;
            }

            card.innerHTML = `
                <div class="card-header">
                    <span class="card-icon">${def.icon}</span>
                    <h3 class="card-title">${def.title}</h3>
                </div>
                <div class="score-bar-container">
                    <div class="score-bar-fill" style="background-color: ${color}; color: ${color}; width: 0%;" data-target="${data.score}%"></div>
                </div>
                <div class="card-content">
                    <span class="card-score-text" style="color: ${color}">Score: ${data.score}%</span>
                    ${contentHTML}
                </div>
            `;
            analysisGrid.appendChild(card);

            setTimeout(() => {
                const bar = card.querySelector('.score-bar-fill');
                if (bar) bar.style.width = bar.getAttribute('data-target');
            }, 300 + (index * 100));

            if (def.key === 'statistical' && data.histogram_data) {
                setTimeout(() => renderHistogram(`chart-${def.key}`, data.histogram_data), 500);
            }
        });
    }

    function renderHistogram(canvasId, data) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        new Chart(canvas, {
            type: 'bar',
            data: {
                labels: Array.from({length: 32}, (_, i) => i * 8),
                datasets: [
                    { label: 'R', data: data.r || [], backgroundColor: 'rgba(239, 68, 68, 0.8)', barPercentage: 1.0, categoryPercentage: 1.0 },
                    { label: 'G', data: data.g || [], backgroundColor: 'rgba(16, 185, 129, 0.8)', barPercentage: 1.0, categoryPercentage: 1.0 },
                    { label: 'B', data: data.b || [], backgroundColor: 'rgba(59, 130, 246, 0.8)', barPercentage: 1.0, categoryPercentage: 1.0 }
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
});
