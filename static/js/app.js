let currentTokens = [];
let selectedTokens = [];
let analysisData = null;
let currentHighlightedLayer = null;

async function tokenizeSentence() {
    const sentence = document.getElementById('sentence-input').value.trim();
    if (!sentence) {
        showError('Please enter a sentence');
        return;
    }

    showLoading(true);
    hideError();

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sentence: sentence,
                token_indices: []
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        currentTokens = data.tokens;
        selectedTokens = [];

        displayTokenSelector();
        document.getElementById('analyze-btn').disabled = true;

    } catch (error) {
        showError('Failed to tokenize sentence: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function displayTokenSelector() {
    const selector = document.getElementById('token-selector');
    selector.innerHTML = '';
    selector.style.display = 'flex';

    currentTokens.forEach((token, index) => {
        const chip = document.createElement('div');
        chip.className = 'token-chip';
        chip.textContent = `${index}: ${token}`;
        chip.onclick = () => toggleToken(index, chip);
        selector.appendChild(chip);
    });

    updateSelectedCount();
}

function toggleToken(index, chipElement) {
    if (selectedTokens.includes(index)) {
        selectedTokens = selectedTokens.filter(i => i !== index);
        chipElement.classList.remove('selected');
        chipElement.removeAttribute('data-order');
    } else if (selectedTokens.length < 6) {
        selectedTokens.push(index);
        chipElement.classList.add('selected');
        chipElement.setAttribute('data-order', selectedTokens.length);
    } else {
        showMessage('Maximum 6 tokens can be selected', 'warning');
    }

    updateSelectedCount();
    document.getElementById('analyze-btn').disabled = selectedTokens.length === 0;
}

function updateSelectedCount() {
    const countEl = document.getElementById('selected-count');
    if (selectedTokens.length === 0) {
        countEl.textContent = 'No tokens selected (select 1-6 tokens)';
        countEl.style.color = '#6c757d';
    } else {
        const tokenNames = selectedTokens.map(i => currentTokens[i]).join(', ');
        countEl.textContent = `Selected: ${tokenNames} (${selectedTokens.length}/6)`;
        countEl.style.color = selectedTokens.length >= 4 ? '#e74c3c' : '#28a745';
    }
}

async function analyzeSentence() {
    if (selectedTokens.length === 0) {
        showError('Please select at least one token');
        return;
    }

    const sentence = document.getElementById('sentence-input').value.trim();
    const reductionMethod = document.getElementById('reduction-method').value;
    showLoading(true);
    hideError();

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sentence: sentence,
                token_indices: selectedTokens,
                reduction_method: reductionMethod
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        analysisData = await response.json();
        displayResults();
        displayModelArchitecture();

    } catch (error) {
        showError('Analysis failed: ' + error.message);
    } finally {
        showLoading(false);
    }
}

function displayModelArchitecture() {
    const treeContainer = document.getElementById('architecture-tree');
    const totalParamsEl = document.getElementById('total-params');

    if (!analysisData || !analysisData.model_architecture) {
        return;
    }

    const totalParams = analysisData.total_parameters;
    totalParamsEl.textContent = `${(totalParams / 1000000).toFixed(1)}M parameters`;

    const grouped = {};
    analysisData.model_architecture.forEach(item => {
        const parts = item.layer_name.split('.');
        const mainComponent = parts[0];

        if (!grouped[mainComponent]) {
            grouped[mainComponent] = [];
        }
        grouped[mainComponent].push(item);
    });

    let html = '';

    if (grouped.embeddings) {
        html += createArchitectureSection('embeddings', grouped.embeddings);
    }

    if (grouped.layers) {
        const layersSummary = grouped.layers.find(item => item.layer_name === 'layers');
        if (layersSummary) {
            html += `
                <div class="arch-item" style="margin-top: 10px;">
                    <span class="layer-name">layers</span>
                    <span class="layer-type">(22x EncoderLayer)</span>
                    <span class="param-count">${formatParamCount(layersSummary.parameters)}</span>
                </div>
            `;
        }

        const layer0Items = grouped.layers.filter(item => 
            item.layer_name.startsWith('layers.0') && item.layer_name !== 'layers'
        );
        html += createLayerDetailedSection(layer0Items);
        html += '<div class="arch-item indent-1" style="color: #7f8c8d; font-style: italic;">... (layers 1-21 have identical structure) ...</div>';
    }

    if (grouped.final_norm) {
        html += createArchitectureSection('final_norm', grouped.final_norm);
    }

    treeContainer.innerHTML = html;
}

function createArchitectureSection(name, items) {
    let html = '';

    items.forEach(item => {
        const indent = item.layer_name.split('.').length - 1;
        const indentClass = `indent-${Math.min(indent, 3)}`;
        const paramCount = formatParamCount(item.parameters);

        let displayName = item.layer_name.split('.').pop();
        if (displayName === 'tok_embeddings') displayName = 'tok_embeddings';
        else if (displayName.includes('norm')) displayName = displayName;
        else if (displayName === 'drop') displayName = 'drop';

        html += `
            <div class="arch-item ${indentClass}" data-layer="${item.layer_name}">
                <div class="layer-info">
                    <span class="layer-name">${displayName}</span>
                    <span class="layer-type">${item.layer_type}</span>
                </div>
                <span class="param-count">${paramCount}</span>
            </div>
        `;
    });

    return html;
}

function createLayerDetailedSection(items) {
    let html = '';

    const mainLayer = items.find(item => item.layer_name === 'layers.0');
    if (mainLayer) {
        html += `
            <div class="arch-item indent-1" data-layer="layers.0">
                <span class="layer-name">[0] EncoderLayer (example)</span>
                <span class="param-count">${formatParamCount(mainLayer.parameters)}</span>
            </div>
        `;
    }

    const components = {};
    items.forEach(item => {
        if (item.layer_name === 'layers.0') return;

        const parts = item.layer_name.split('.');
        if (parts.length === 3) {
            const compName = parts[2];
            if (!components[compName]) components[compName] = [];
            components[compName].push(item);
        } else if (parts.length === 4) {
            const parentComp = parts[2];
            if (!components[parentComp]) components[parentComp] = [];
            components[parentComp].push(item);
        }
    });

    const componentOrder = ['attn_norm', 'attn', 'mlp_norm', 'mlp'];
    componentOrder.forEach(compName => {
        if (components[compName]) {
            components[compName].forEach(item => {
                const parts = item.layer_name.split('.');
                const isSubComponent = parts.length === 4;
                const indentClass = isSubComponent ? 'indent-3' : 'indent-2';

                let displayName = parts[parts.length - 1];
                if (compName === 'attn' && !isSubComponent) {
                    displayName = 'attention';
                } else if (compName === 'mlp' && !isSubComponent) {
                    displayName = 'mlp';
                } else if (displayName.includes('norm')) {
                    displayName = displayName;
                } else if (displayName === 'Wqkv') {
                    displayName = 'Wqkv (Q,K,V projection)';
                } else if (displayName === 'Wo') {
                    displayName = 'Wo (output projection)';
                } else if (displayName === 'Wi') {
                    displayName = 'Wi (input projection)';
                } else if (displayName === 'rotary_emb') {
                    displayName = 'rotary_emb';
                }

                const isAttention = compName === 'attn' && !isSubComponent;
                const attentionClass = isAttention ? 'attention-layer' : '';

                html += `
                    <div class="arch-item ${indentClass} ${attentionClass}" data-layer="${item.layer_name}">
                        <div class="layer-info">
                            <span class="layer-name">${displayName}</span>
                        </div>
                        <span class="param-count">${formatParamCount(item.parameters)}</span>
                    </div>
                `;
            });
        }
    });

    return html;
}

function formatParamCount(count) {
    if (count >= 1000000) {
        return `${(count / 1000000).toFixed(1)}M`;
    } else if (count >= 1000) {
        return `${(count / 1000).toFixed(1)}K`;
    } else {
        return count.toString();
    }
}

function highlightArchitectureLayer(layerIndex) {
    document.querySelectorAll('.arch-item.highlighted').forEach(el => {
        el.classList.remove('highlighted');
    });

    const layerElement = document.querySelector(`[data-layer="layers.${layerIndex}"]`);
    if (layerElement) {
        layerElement.classList.add('highlighted');
        layerElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    currentHighlightedLayer = layerIndex;
}

function showMessage(message, type = 'info') {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${type}`;
    messageEl.textContent = message;
    messageEl.style.cssText = `
        position: fixed; 
        top: 20px; 
        right: 20px; 
        padding: 12px 20px; 
        border-radius: 6px; 
        z-index: 1000;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateX(100%);
        transition: transform 0.3s ease;
        ${type === 'warning' ? 'background: #fff3cd; color: #856404; border: 1px solid #ffeaa7;' : 'background: #d4edda; color: #155724; border: 1px solid #c3e6cb;'}
    `;

    document.body.appendChild(messageEl);

    setTimeout(() => {
        messageEl.style.transform = 'translateX(0)';
    }, 100);

    setTimeout(() => {
        messageEl.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (messageEl.parentNode) {
                messageEl.parentNode.removeChild(messageEl);
            }
        }, 300);
    }, 3000);
}

function displayResults() {
    document.getElementById('results').style.display = 'block';
    plotTrajectories();
    setupAttentionControls();
    updateAttentionPlot();
}

function plotTrajectories() {
    const colorPalette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
    ];

    // Define method labels early so they can be used in hovertemplate
    const methodLabels = {
        'pca': ['PCA Component 1', 'PCA Component 2'],
        'tsne': ['t-SNE Dimension 1', 't-SNE Dimension 2']
    };
    const method = analysisData.reduction_method || 'pca';
    const [xLabel, yLabel] = methodLabels[method] || methodLabels.pca;

    const traces = analysisData.trajectories.map((traj, index) => {
        const color = colorPalette[index % colorPalette.length];

        const x = traj.trajectory.map(point => point[0]);
        const y = traj.trajectory.map(point => point[1]);

        const layerLabels = x.map((_, layerIdx) => `Layer ${layerIdx}`);

        return {
            x: x,
            y: y,
            mode: 'lines+markers',
            name: `${traj.token} (${traj.token_index})`,
            line: { 
                color: color, 
                width: 3,
                shape: 'spline',
                smoothing: 0.3
            },
            marker: { 
                size: 8, 
                color: color,
                line: { color: 'white', width: 2 },
                symbol: index % 2 === 0 ? 'circle' : 'square'
            },
            text: layerLabels,
            hovertemplate: `<b>${traj.token}</b> (idx: ${traj.token_index})<br>%{text}<br>${xLabel}: %{x:.3f}<br>${yLabel}: %{y:.3f}<extra></extra>`
        };
    });

    const annotations = [];
    analysisData.trajectories.forEach((traj, index) => {
        const color = colorPalette[index % colorPalette.length];
        const x = traj.trajectory.map(point => point[0]);
        const y = traj.trajectory.map(point => point[1]);

        if (x.length > 0) {
            annotations.push({
                x: x[0],
                y: y[0],
                text: 'START',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: color,
                font: { size: 10, color: color },
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: color,
                borderwidth: 1
            });

            annotations.push({
                x: x[x.length - 1],
                y: y[y.length - 1],
                text: 'END',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: color,
                font: { size: 10, color: color },
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: color,
                borderwidth: 1
            });
        }
    });

    const layout = {
        xaxis: { 
            title: xLabel,
            gridcolor: '#e0e0e0',
            zerolinecolor: '#d0d0d0'
        },
        yaxis: { 
            title: yLabel,
            gridcolor: '#e0e0e0',
            zerolinecolor: '#d0d0d0'
        },
        hovermode: 'closest',
        showlegend: true,
        legend: { 
            x: 0, 
            y: 1,
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#d0d0d0',
            borderwidth: 1
        },
        margin: { t: 30, b: 50, l: 60, r: 30 },
        plot_bgcolor: '#fafafa',
        annotations: annotations
    };

    Plotly.newPlot('trajectory-plot', traces, layout, {responsive: true});

    const variance = analysisData.explained_variance;
    let varianceText = '';
    
    if (method === 'pca') {
        const variancePercent = (variance * 100).toFixed(1);
        varianceText = `PCA explains <strong>${variancePercent}%</strong> of variance`;
    } else if (method === 'tsne') {
        varianceText = `t-SNE KL divergence: <strong>${variance.toFixed(3)}</strong>`;
    }
    
    document.getElementById('variance-info').innerHTML = 
        `${varianceText} | Showing <strong>${analysisData.trajectories.length}</strong> token trajectories across <strong>${analysisData.trajectories[0]?.trajectory.length || 0}</strong> layers`;
}

function setupAttentionControls() {
    const layerSelect = document.getElementById('layer-select');
    layerSelect.innerHTML = '';

    analysisData.attention_data.forEach(data => {
        const option = document.createElement('option');
        option.value = data.layer;
        option.textContent = `Layer ${data.layer}`;
        layerSelect.appendChild(option);
    });
}

function updateAttentionPlot() {
    const selectedLayer = parseInt(document.getElementById('layer-select').value);
    const layerData = analysisData.attention_data.find(d => d.layer === selectedLayer);

    if (!layerData) return;

    highlightArchitectureLayer(selectedLayer);

    const trace = {
        z: layerData.attention_matrix,
        x: layerData.tokens,
        y: layerData.tokens,
        type: 'heatmap',
        colorscale: 'Viridis',
        hovertemplate: 'From: %{y}<br>To: %{x}<br>Attention: %{z:.3f}<extra></extra>'
    };

    const layout = {
        xaxis: { 
            title: 'To Token',
            tickangle: 45,
            side: 'bottom'
        },
        yaxis: { 
            title: 'From Token',
            autorange: 'reversed'
        },
        margin: { t: 30, b: 100, l: 100, r: 30 }
    };

    Plotly.newPlot('attention-plot', [trace], layout, {responsive: true});
}

function toggleArchitecture() {
    const panel = document.querySelector('.architecture-panel');
    panel.classList.toggle('mobile-visible');
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

function showError(message) {
    const errorEl = document.getElementById('error');
    errorEl.textContent = message;
    errorEl.style.display = 'block';
}

function hideError() {
    document.getElementById('error').style.display = 'none';
}

window.onload = function() {
    tokenizeSentence();
};