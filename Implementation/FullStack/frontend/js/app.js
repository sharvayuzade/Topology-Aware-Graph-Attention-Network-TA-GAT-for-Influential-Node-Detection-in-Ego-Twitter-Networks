/**
 * TA-GAT Studio - Enterprise Graph Intelligence Frontend
 * Architecture: Modular State Management, Interactive Vis.js Engine,
 * Diffusion Outbreak Player, Paginated Table & Multi-Chart Analytics.
 */

// ============================================================================
// 1. Toast Notification Service
// ============================================================================
class ToastManager {
  static show(message, type = 'info', duration = 3500) {
    let container = document.getElementById('toastContainer');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toastContainer';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = 'custom-toast';
    
    let icon = 'fa-info-circle text-primary';
    if (type === 'success') icon = 'fa-check-circle text-success';
    if (type === 'warning') icon = 'fa-exclamation-triangle text-warning';
    if (type === 'error') icon = 'fa-times-circle text-danger';

    toast.innerHTML = `
      <i class="fas ${icon} fa-lg"></i>
      <div class="flex-grow-1 small fw-medium">${message}</div>
      <button type="button" class="btn-close btn-close-white small ms-2" aria-label="Close"></button>
    `;

    const closeBtn = toast.querySelector('.btn-close');
    closeBtn.addEventListener('click', () => toast.remove());

    container.appendChild(toast);

    setTimeout(() => {
      if (toast.parentElement) {
        toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(50px)';
        setTimeout(() => toast.remove(), 300);
      }
    }, duration);
  }
}

// ============================================================================
// 2. Theme Manager (Dark / Light Mode)
// ============================================================================
class ThemeManager {
  constructor() {
    this.theme = localStorage.getItem('tagat_theme') || 'dark';
    this.toggleBtn = document.getElementById('themeToggleBtn');
    this.init();
  }

  init() {
    this.applyTheme(this.theme);
    if (this.toggleBtn) {
      this.toggleBtn.addEventListener('click', () => {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        this.applyTheme(this.theme);
        localStorage.setItem('tagat_theme', this.theme);
      });
    }
  }

  applyTheme(theme) {
    document.documentElement.setAttribute('data-bs-theme', theme);
    if (this.toggleBtn) {
      const icon = this.toggleBtn.querySelector('i');
      if (icon) {
        icon.className = theme === 'dark' ? 'fas fa-sun text-warning' : 'fas fa-moon text-primary';
      }
    }
    // Update chart colors if charts exist
    if (window.app && window.app.chartManager) {
      window.app.chartManager.updateThemeColors(theme);
    }
    if (window.app && window.app.networkGraph) {
      window.app.networkGraph.updateTheme(theme);
    }
  }

  get isDark() {
    return this.theme === 'dark';
  }
}

// ============================================================================
// 3. Central Application State
// ============================================================================
const AppState = {
  activeDatasetName: 'Stanford Twitter Ego Network',
  rawPredictions: [],
  filteredPredictions: [],
  visualizationPayload: null,
  diffusionData: null,
  selectedNode: null,
  activeFilterTier: 'ALL',
  searchQuery: '',
  sortKey: 'rank',
  sortAsc: true,
  page: 1,
  pageSize: 15
};

// ============================================================================
// 4. API Client
// ============================================================================
class ApiClient {
  static async checkHealth() {
    const res = await fetch('/api/health');
    return await res.json();
  }

  static async getSamples() {
    const res = await fetch('/api/samples');
    return await res.json();
  }

  static async getModelInfo() {
    const res = await fetch('/api/model_info');
    return await res.json();
  }

  static async predictGraph(datasetSubdir) {
    const res = await fetch('/predict_graph', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_subdir: datasetSubdir })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to analyze ego graph');
    }
    return await res.json();
  }

  static async predictCsv(csvPath) {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ csv_path: csvPath })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to analyze CSV');
    }
    return await res.json();
  }

  static async uploadDataset(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/api/predict_upload', {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to upload dataset');
    }
    return await res.json();
  }

  static async simulateDiffusion(seeds, p, maxSteps) {
    const res = await fetch('/api/simulate_diffusion', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seeds, p, max_steps: maxSteps })
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Diffusion simulation failed');
    }
    return await res.json();
  }
}

// ============================================================================
// 5. Vis.js Network Graph Controller
// ============================================================================
class NetworkGraphController {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.network = null;
    this.nodesDataSet = null;
    this.edgesDataSet = null;
    this.physicsEnabled = true;
    this.isFullscreen = false;
    this.originalNodes = [];
    this.originalEdges = [];
    this.setupHUDControls();
  }

  setupHUDControls() {
    // Search input
    const searchInput = document.getElementById('networkSearchInput');
    if (searchInput) {
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const query = searchInput.value.trim();
          if (query) this.focusNode(query);
        }
      });
    }

    // Freeze Physics Button
    const freezeBtn = document.getElementById('hudFreezeBtn');
    if (freezeBtn) {
      freezeBtn.addEventListener('click', () => this.togglePhysics());
    }

    // Reset Zoom Button
    const fitBtn = document.getElementById('hudFitBtn');
    if (fitBtn) {
      fitBtn.addEventListener('click', () => {
        if (this.network) this.network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
      });
    }

    // Fullscreen Toggle
    const fsBtn = document.getElementById('hudFullscreenBtn');
    if (fsBtn) {
      fsBtn.addEventListener('click', () => this.toggleFullscreen());
    }

    // Tier Filter Selector
    const tierSelect = document.getElementById('hudTierFilter');
    if (tierSelect) {
      tierSelect.addEventListener('change', (e) => {
        this.filterByTier(e.target.value);
      });
    }
  }

  render(vizData, isDark = true) {
    if (!this.container) return;
    if (!vizData || !vizData.nodes || vizData.nodes.length === 0) {
      this.container.innerHTML = `
        <div class="h-100 d-flex flex-column align-items-center justify-content-center text-muted p-4">
          <i class="fas fa-project-diagram fa-3x mb-3 opacity-50"></i>
          <span class="fw-semibold">No direct graph topology available</span>
          <small class="text-dim">Tabular node attributes loaded without edge adjacency.</small>
        </div>
      `;
      return;
    }

    this.originalNodes = JSON.parse(JSON.stringify(vizData.nodes));
    this.originalEdges = JSON.parse(JSON.stringify(vizData.edges || []));

    const styledNodes = this.styleNodes(this.originalNodes, isDark);
    const styledEdges = this.styleEdges(this.originalEdges, isDark);

    this.nodesDataSet = new vis.DataSet(styledNodes);
    this.edgesDataSet = new vis.DataSet(styledEdges);

    const data = { nodes: this.nodesDataSet, edges: this.edgesDataSet };

    const options = {
      nodes: {
        shape: 'dot',
        scaling: { min: 12, max: 36 },
        borderWidth: 2,
        shadow: {
          enabled: isDark,
          color: 'rgba(0,0,0,0.5)',
          size: 6,
          x: 2,
          y: 2
        }
      },
      edges: {
        smooth: { type: 'continuous' },
        arrows: { to: { enabled: true, scaleFactor: 0.55 } },
        selectionWidth: 2.5
      },
      physics: {
        enabled: this.physicsEnabled,
        stabilization: { iterations: 120 },
        barnesHut: {
          gravitationalConstant: -3200,
          centralGravity: 0.35,
          springLength: 95,
          springConstant: 0.045,
          damping: 0.09
        }
      },
      interaction: {
        hover: true,
        tooltipDelay: 80,
        zoomView: true,
        hoverConnectedEdges: true
      }
    };

    if (this.network) {
      this.network.destroy();
    }

    this.network = new vis.Network(this.container, data, options);

    // Node click handler
    this.network.on('click', (params) => {
      if (params.nodes.length > 0) {
        const clickedId = params.nodes[0];
        const record = AppState.rawPredictions.find(p => String(p.node_id) === String(clickedId));
        if (record) {
          window.app.showNodeTelemetry(record);
        }
      }
    });

    // Double click to center
    this.network.on('doubleClick', (params) => {
      if (params.nodes.length > 0) {
        this.focusNode(params.nodes[0]);
      }
    });
  }

  styleNodes(nodes, isDark) {
    return nodes.map(n => {
      let bg = isDark ? '#475569' : '#94a3b8';
      let border = isDark ? '#64748b' : '#64748b';
      let fontColor = isDark ? '#f8fafc' : '#0f172a';
      let size = 13;

      if (n.rank <= 3) {
        bg = '#f59e0b';
        border = '#fbbf24';
        size = 30;
      } else if (n.rank <= 10) {
        bg = '#6366f1';
        border = '#818cf8';
        size = 23;
      } else if (n.rank <= 25) {
        bg = '#06b6d4';
        border = '#22d3ee';
        size = 17;
      }

      return {
        ...n,
        size: size,
        color: {
          background: bg,
          border: border,
          highlight: { background: '#ffffff', border: bg }
        },
        font: {
          color: fontColor,
          size: 11,
          face: 'JetBrains Mono',
          strokeWidth: isDark ? 2 : 3,
          strokeColor: isDark ? '#0b0f19' : '#ffffff'
        }
      };
    });
  }

  styleEdges(edges, isDark) {
    const edgeColor = isDark ? 'rgba(148, 163, 184, 0.25)' : 'rgba(100, 116, 139, 0.28)';
    const highlightColor = isDark ? '#6366f1' : '#4f46e5';

    return edges.map(e => ({
      ...e,
      color: {
        color: edgeColor,
        highlight: highlightColor,
        hover: highlightColor
      },
      width: 1.2
    }));
  }

  updateTheme(theme) {
    if (!this.network || !this.originalNodes.length) return;
    const isDark = theme === 'dark';
    const styledNodes = this.styleNodes(this.originalNodes, isDark);
    const styledEdges = this.styleEdges(this.originalEdges, isDark);
    this.nodesDataSet.clear();
    this.nodesDataSet.add(styledNodes);
    this.edgesDataSet.clear();
    this.edgesDataSet.add(styledEdges);
  }

  focusNode(nodeId) {
    if (!this.network || !this.nodesDataSet) return;
    const match = this.nodesDataSet.get(nodeId) || this.nodesDataSet.get(parseInt(nodeId));
    if (!match) {
      ToastManager.show(`Node ID "${nodeId}" not found in current graph sample.`, 'warning');
      return;
    }

    this.network.selectNodes([match.id]);
    this.network.focus(match.id, {
      scale: 1.6,
      animation: {
        duration: 900,
        easingFunction: 'easeInOutCubic'
      }
    });

    const record = AppState.rawPredictions.find(p => String(p.node_id) === String(match.id));
    if (record) {
      ToastManager.show(`Focused Node #${record.rank}: ID ${record.node_id} (Score: ${record.pred})`, 'info');
    }
  }

  filterByTier(tier) {
    if (!this.nodesDataSet || !this.originalNodes) return;
    AppState.activeFilterTier = tier;

    let filtered = this.originalNodes;
    if (tier === 'TOP3') {
      filtered = this.originalNodes.filter(n => n.rank <= 3);
    } else if (tier === 'TOP10') {
      filtered = this.originalNodes.filter(n => n.rank <= 10);
    } else if (tier === 'TOP25') {
      filtered = this.originalNodes.filter(n => n.rank <= 25);
    }

    const isDark = document.documentElement.getAttribute('data-bs-theme') !== 'light';
    this.nodesDataSet.clear();
    this.nodesDataSet.add(this.styleNodes(filtered, isDark));
    this.network.fit({ animation: { duration: 500 } });
  }

  togglePhysics() {
    if (!this.network) return;
    this.physicsEnabled = !this.physicsEnabled;
    this.network.setOptions({ physics: { enabled: this.physicsEnabled } });
    const btn = document.getElementById('hudFreezeBtn');
    if (btn) {
      btn.innerHTML = this.physicsEnabled 
        ? '<i class="fas fa-snowflake me-1"></i> Freeze' 
        : '<i class="fas fa-play me-1"></i> Run Physics';
    }
    ToastManager.show(this.physicsEnabled ? 'Physics simulation active' : 'Physics simulation paused', 'info');
  }

  toggleFullscreen() {
    const wrapper = document.querySelector('.network-container-wrapper');
    if (!wrapper) return;
    this.isFullscreen = !this.isFullscreen;
    wrapper.classList.toggle('fullscreen-mode', this.isFullscreen);

    const btn = document.getElementById('hudFullscreenBtn');
    if (btn) {
      btn.innerHTML = this.isFullscreen ? '<i class="fas fa-compress"></i>' : '<i class="fas fa-expand"></i>';
    }

    setTimeout(() => {
      if (this.network) this.network.fit();
    }, 200);
  }

  highlightCascadeStep(newlyActiveIds, allActiveIds) {
    if (!this.nodesDataSet) return;

    const updates = [];
    allActiveIds.forEach(id => {
      const isNew = newlyActiveIds.includes(id);
      updates.push({
        id: id,
        color: {
          background: isNew ? '#ef4444' : '#f59e0b',
          border: isNew ? '#ff0055' : '#fbbf24',
          highlight: { background: '#ffffff', border: '#ef4444' }
        },
        size: isNew ? 28 : 22
      });
    });

    this.nodesDataSet.update(updates);
  }

  resetColors() {
    if (!this.nodesDataSet || !this.originalNodes.length) return;
    const isDark = document.documentElement.getAttribute('data-bs-theme') !== 'light';
    this.nodesDataSet.clear();
    this.nodesDataSet.add(this.styleNodes(this.originalNodes, isDark));
  }
}

// ============================================================================
// 6. Interactive Diffusion Cascade Player
// ============================================================================
class DiffusionPlayer {
  constructor(networkGraph, chartManager) {
    this.networkGraph = networkGraph;
    this.chartManager = chartManager;
    this.data = null;
    this.currentStep = 0;
    this.isPlaying = false;
    this.timer = null;
    this.stepSpeed = 800; // ms

    this.bindControls();
  }

  bindControls() {
    // Sliders
    const probRange = document.getElementById('probRange');
    const probVal = document.getElementById('probVal');
    if (probRange && probVal) {
      probRange.addEventListener('input', (e) => {
        probVal.textContent = parseFloat(e.target.value).toFixed(2);
      });
    }

    const stepsRange = document.getElementById('stepsRange');
    const stepsVal = document.getElementById('stepsVal');
    if (stepsRange && stepsVal) {
      stepsRange.addEventListener('input', (e) => {
        stepsVal.textContent = e.target.value;
      });
    }

    // Run Simulation
    const runBtn = document.getElementById('runSimulationBtn');
    if (runBtn) {
      runBtn.addEventListener('click', () => this.runSimulation());
    }

    // Playback Buttons
    document.getElementById('cascadePlayBtn')?.addEventListener('click', () => this.togglePlay());
    document.getElementById('cascadeStepForwardBtn')?.addEventListener('click', () => this.stepForward());
    document.getElementById('cascadeStepBackBtn')?.addEventListener('click', () => this.stepBackward());
    document.getElementById('cascadeResetBtn')?.addEventListener('click', () => this.resetPlayback());

    // Speed Selector
    document.getElementById('cascadeSpeedSelect')?.addEventListener('change', (e) => {
      this.stepSpeed = parseInt(e.target.value, 10);
      if (this.isPlaying) {
        this.pause();
        this.play();
      }
    });
  }

  async runSimulation() {
    const p = parseFloat(document.getElementById('probRange').value);
    const maxSteps = parseInt(document.getElementById('stepsRange').value);
    const strat = document.getElementById('seedStrategySelect').value;

    let seeds = [];
    if (strat === 'single_top') {
      seeds = AppState.rawPredictions.slice(0, 1).map(n => n.node_id);
    } else if (strat === 'top5') {
      seeds = AppState.rawPredictions.slice(0, 5).map(n => n.node_id);
    } else {
      seeds = AppState.rawPredictions.slice(0, 3).map(n => n.node_id);
    }

    try {
      ToastManager.show('Simulating Independent Cascade propagation...', 'info');
      const data = await ApiClient.simulateDiffusion(seeds, p, maxSteps);
      this.data = data;
      AppState.diffusionData = data;
      this.currentStep = 0;

      // Reveal Results Deck
      document.getElementById('diffusionResultsDeck').style.display = 'block';
      document.getElementById('diffReachedNodes').textContent = data.total_reached;
      document.getElementById('diffReachPct').textContent = `${data.reach_percentage}% Reach`;
      document.getElementById('diffSteps').textContent = data.total_steps;
      document.getElementById('diffVelocity').textContent = data.cascade_velocity;

      // Render Wave Badges
      this.renderWaveBadges();

      // Render Diffusion S-Curve Chart
      if (this.chartManager) {
        this.chartManager.renderDiffusionCurve(data.waves);
      }

      // Reset Network & Jump to Step 0
      this.jumpToStep(0);
      ToastManager.show(`Simulation Complete! Reached ${data.total_reached} nodes across ${data.total_steps} waves.`, 'success');
      
      // Auto-scroll to results
      document.getElementById('diffusionResultsDeck').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (err) {
      ToastManager.show(`Simulation error: ${err.message}`, 'error');
    }
  }

  renderWaveBadges() {
    const container = document.getElementById('diffusionWaveContainer');
    if (!container || !this.data) return;

    let html = '';
    this.data.waves.forEach((w, idx) => {
      html += `
        <div class="diffusion-wave-pill ${idx === 0 ? 'active-step' : ''}" id="wave-pill-${w.step}" onclick="window.app.diffusionPlayer.jumpToStep(${w.step})">
          <strong>Wave ${w.step}:</strong> +${w.newly_activated.length} nodes 
          <span class="badge bg-dark ms-1">Total: ${w.total_active}</span>
        </div>
      `;
    });
    container.innerHTML = html;
  }

  jumpToStep(step) {
    if (!this.data || !this.data.waves || step < 0 || step >= this.data.waves.length) return;
    this.currentStep = step;

    // Update Counter HUD
    const counter = document.getElementById('cascadeStepCounter');
    if (counter) counter.textContent = `Step ${step} / ${this.data.total_steps}`;

    // Highlight active wave pill
    document.querySelectorAll('.diffusion-wave-pill').forEach((pill, idx) => {
      pill.classList.toggle('active-step', idx === step);
    });

    // Accumulate all active nodes up to this step
    const allActive = new Set();
    for (let i = 0; i <= step; i++) {
      this.data.waves[i].newly_activated.forEach(n => allActive.add(n));
    }
    const newlyActive = this.data.waves[step].newly_activated;

    // Trigger visual flare on Vis.js network
    this.networkGraph.highlightCascadeStep(newlyActive, Array.from(allActive));
  }

  stepForward() {
    if (!this.data) return;
    if (this.currentStep < this.data.waves.length - 1) {
      this.jumpToStep(this.currentStep + 1);
    } else {
      this.pause();
    }
  }

  stepBackward() {
    if (!this.data) return;
    if (this.currentStep > 0) {
      this.jumpToStep(this.currentStep - 1);
    }
  }

  resetPlayback() {
    this.pause();
    this.jumpToStep(0);
    this.networkGraph.resetColors();
  }

  togglePlay() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  play() {
    if (!this.data) return;
    this.isPlaying = true;
    const playBtn = document.getElementById('cascadePlayBtn');
    if (playBtn) playBtn.innerHTML = '<i class="fas fa-pause"></i> Pause';

    if (this.currentStep >= this.data.waves.length - 1) {
      this.currentStep = 0;
    }

    this.timer = setInterval(() => {
      if (this.currentStep < this.data.waves.length - 1) {
        this.stepForward();
      } else {
        this.pause();
      }
    }, this.stepSpeed);
  }

  pause() {
    this.isPlaying = false;
    if (this.timer) clearInterval(this.timer);
    const playBtn = document.getElementById('cascadePlayBtn');
    if (playBtn) playBtn.innerHTML = '<i class="fas fa-play"></i> Play Outbreak';
  }
}

// ============================================================================
// 7. Paginated & Sortable Rankings Table Controller
// ============================================================================
class RankingsTableController {
  constructor() {
    this.tbody = document.getElementById('rankingsTableBody');
    this.paginationContainer = document.getElementById('tablePagination');
    this.counterEl = document.getElementById('tableCounter');
    this.pageSizeSelect = document.getElementById('tablePageSizeSelect');
    this.searchInput = document.getElementById('searchNodeInput');
    
    this.setupListeners();
  }

  setupListeners() {
    // Search input
    if (this.searchInput) {
      this.searchInput.addEventListener('input', (e) => {
        AppState.searchQuery = e.target.value.trim().toLowerCase();
        AppState.page = 1;
        this.render();
      });
    }

    // Page size
    if (this.pageSizeSelect) {
      this.pageSizeSelect.addEventListener('change', (e) => {
        AppState.pageSize = parseInt(e.target.value, 10);
        AppState.page = 1;
        this.render();
      });
    }

    // Table Header Sorting
    const sortHeaders = document.querySelectorAll('.table-enterprise th[data-sort]');
    sortHeaders.forEach(th => {
      th.addEventListener('click', () => {
        const key = th.getAttribute('data-sort');
        if (AppState.sortKey === key) {
          AppState.sortAsc = !AppState.sortAsc;
        } else {
          AppState.sortKey = key;
          AppState.sortAsc = key === 'rank'; // rank defaults ascending, scores descending
        }
        this.updateSortHeaderIcons();
        this.render();
      });
    });

    // Quick Filter Chips
    const chips = document.querySelectorAll('.filter-chip[data-filter]');
    chips.forEach(chip => {
      chip.addEventListener('click', () => {
        chips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        AppState.activeFilterTier = chip.getAttribute('data-filter');
        AppState.page = 1;
        this.render();
      });
    });
  }

  updateSortHeaderIcons() {
    document.querySelectorAll('.table-enterprise th[data-sort]').forEach(th => {
      const key = th.getAttribute('data-sort');
      const icon = th.querySelector('.sort-icon');
      if (!icon) return;

      if (AppState.sortKey === key) {
        th.classList.add('sort-active');
        icon.className = AppState.sortAsc ? 'fas fa-sort-up ms-1 sort-icon' : 'fas fa-sort-down ms-1 sort-icon';
      } else {
        th.classList.remove('sort-active');
        icon.className = 'fas fa-sort ms-1 opacity-25 sort-icon';
      }
    });
  }

  getFilteredSortedData() {
    let data = [...AppState.rawPredictions];

    // Tier Filter
    if (AppState.activeFilterTier === 'TOP3') {
      data = data.filter(d => d.rank <= 3);
    } else if (AppState.activeFilterTier === 'TOP10') {
      data = data.filter(d => d.rank <= 10);
    } else if (AppState.activeFilterTier === 'TOP25') {
      data = data.filter(d => d.rank <= 25);
    }

    // Text Search
    if (AppState.searchQuery) {
      data = data.filter(d => String(d.node_id).toLowerCase().includes(AppState.searchQuery));
    }

    // Sorting
    data.sort((a, b) => {
      let valA = a[AppState.sortKey];
      let valB = b[AppState.sortKey];
      if (valA === undefined) valA = 0;
      if (valB === undefined) valB = 0;

      if (typeof valA === 'string') {
        return AppState.sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return AppState.sortAsc ? valA - valB : valB - valA;
    });

    return data;
  }

  render() {
    if (!this.tbody) return;
    const data = this.getFilteredSortedData();
    const totalRecords = data.length;
    const totalPages = Math.ceil(totalRecords / AppState.pageSize) || 1;

    if (AppState.page > totalPages) AppState.page = totalPages;
    if (AppState.page < 1) AppState.page = 1;

    const startIndex = (AppState.page - 1) * AppState.pageSize;
    const pagedData = data.slice(startIndex, startIndex + AppState.pageSize);

    if (this.counterEl) {
      const end = Math.min(startIndex + AppState.pageSize, totalRecords);
      this.counterEl.textContent = totalRecords > 0 
        ? `Showing ${startIndex + 1}–${end} of ${totalRecords} nodes`
        : '0 nodes found';
    }

    if (pagedData.length === 0) {
      this.tbody.innerHTML = `
        <tr>
          <td colspan="8" class="text-center text-muted py-4">
            <i class="fas fa-search me-2"></i> No matching nodes found for the given criteria.
          </td>
        </tr>
      `;
      this.renderPagination(totalPages);
      return;
    }

    const maxScore = Math.max(...AppState.rawPredictions.map(p => p.pred || 1));

    let html = '';
    pagedData.forEach((row) => {
      let rankClass = 'rank-default';
      if (row.rank === 1) rankClass = 'rank-gold';
      else if (row.rank === 2) rankClass = 'rank-silver';
      else if (row.rank === 3) rankClass = 'rank-bronze';

      const score = row.pred !== undefined ? parseFloat(row.pred).toFixed(4) : '0.0000';
      const pct = Math.min(100, Math.max(8, (parseFloat(score) / maxScore) * 100)).toFixed(0);

      html += `
        <tr onclick="window.app.onTableRowClick('${row.node_id}')">
          <td><span class="rank-badge ${rankClass}">#${row.rank}</span></td>
          <td>
            <div class="d-flex align-items-center gap-2">
              <code class="fw-bold" style="color: var(--accent-indigo); font-size: 0.92rem;">${row.node_id}</code>
              <button class="btn btn-sm p-0 text-muted opacity-50" title="Focus Node" onclick="event.stopPropagation(); window.app.networkGraph.focusNode('${row.node_id}')">
                <i class="fas fa-crosshairs"></i>
              </button>
            </div>
          </td>
          <td>
            <div class="d-flex align-items-center gap-2">
              <div class="score-track">
                <div class="score-fill" style="width: ${pct}%;"></div>
              </div>
              <span class="font-mono fw-bold" style="font-size: 0.88rem;">${score}</span>
            </div>
          </td>
          <td class="fw-semibold">${row.degree !== undefined ? row.degree : '-'}</td>
          <td>
            <span class="fw-semibold text-heading">${row.in_degree !== undefined ? row.in_degree : '-'}</span> / 
            <span class="text-dim">${row.out_degree !== undefined ? row.out_degree : '-'}</span>
          </td>
          <td class="font-mono text-muted" style="font-size: 0.82rem;">${row.pagerank !== undefined ? parseFloat(row.pagerank).toFixed(5) : '-'}</td>
          <td class="font-mono text-muted" style="font-size: 0.82rem;">${row.betweenness !== undefined ? parseFloat(row.betweenness).toFixed(5) : '-'}</td>
          <td><span class="badge bg-dark-subtle border text-dim font-mono">${row.core_number !== undefined ? row.core_number : '-'}</span></td>
        </tr>
      `;
    });

    this.tbody.innerHTML = html;
    this.renderPagination(totalPages);
  }

  renderPagination(totalPages) {
    if (!this.paginationContainer) return;
    if (totalPages <= 1) {
      this.paginationContainer.innerHTML = '';
      return;
    }

    let html = `
      <li class="page-item ${AppState.page === 1 ? 'disabled' : ''}">
        <a class="page-link" href="#" onclick="event.preventDefault(); window.app.rankingsTable.goToPage(1)">&laquo;</a>
      </li>
      <li class="page-item ${AppState.page === 1 ? 'disabled' : ''}">
        <a class="page-link" href="#" onclick="event.preventDefault(); window.app.rankingsTable.goToPage(${AppState.page - 1})">&lsaquo;</a>
      </li>
    `;

    const maxVisible = 5;
    let startPage = Math.max(1, AppState.page - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);
    if (endPage - startPage + 1 < maxVisible) {
      startPage = Math.max(1, endPage - maxVisible + 1);
    }

    for (let p = startPage; p <= endPage; p++) {
      html += `
        <li class="page-item ${p === AppState.page ? 'active' : ''}">
          <a class="page-link" href="#" onclick="event.preventDefault(); window.app.rankingsTable.goToPage(${p})">${p}</a>
        </li>
      `;
    }

    html += `
      <li class="page-item ${AppState.page === totalPages ? 'disabled' : ''}">
        <a class="page-link" href="#" onclick="event.preventDefault(); window.app.rankingsTable.goToPage(${AppState.page + 1})">&rsaquo;</a>
      </li>
      <li class="page-item ${AppState.page === totalPages ? 'disabled' : ''}">
        <a class="page-link" href="#" onclick="event.preventDefault(); window.app.rankingsTable.goToPage(${totalPages})">&raquo;</a>
      </li>
    `;

    this.paginationContainer.innerHTML = html;
  }

  goToPage(p) {
    AppState.page = p;
    this.render();
  }
}

// ============================================================================
// 8. Chart Manager (Top 10 Influencers, Cascade S-Curve, Node Radar)
// ============================================================================
class ChartManager {
  constructor() {
    this.barChart = null;
    this.diffusionChart = null;
    this.radarChart = null;
  }

  renderBarChart(topNodes, isDark = true) {
    const canvas = document.getElementById('rankingChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    if (this.barChart) this.barChart.destroy();

    const labels = topNodes.map((n, i) => `#${i + 1} Node ${n.node_id}`);
    const scores = topNodes.map(n => n.pred);

    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.06)';

    this.barChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Influence Score',
          data: scores,
          backgroundColor: [
            '#f59e0b', '#fbbf24', '#fcd34d',
            '#6366f1', '#818cf8', '#a5b4fc',
            '#06b6d4', '#22d3ee', '#67e8f9',
            '#10b981'
          ],
          borderRadius: 6,
          borderWidth: 0
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (c) => ` TA-GAT Predicted Score: ${parseFloat(c.raw).toFixed(4)}`
            }
          }
        },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            grid: { display: false },
            ticks: { color: isDark ? '#f8fafc' : '#0f172a', font: { family: 'Outfit', size: 11, weight: '600' } }
          }
        }
      }
    });
  }

  renderDiffusionCurve(waves, isDark = true) {
    const canvas = document.getElementById('diffusionCurveChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    if (this.diffusionChart) this.diffusionChart.destroy();

    const labels = waves.map(w => `Step ${w.step}`);
    const cumulative = waves.map(w => w.total_active);
    const waveInc = waves.map(w => w.newly_activated.length);

    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.06)';

    this.diffusionChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Cumulative Active Nodes',
            data: cumulative,
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.15)',
            tension: 0.35,
            fill: true,
            pointRadius: 4,
            pointBackgroundColor: '#818cf8',
            yAxisID: 'y'
          },
          {
            label: 'New Outbreak per Wave',
            data: waveInc,
            type: 'bar',
            backgroundColor: 'rgba(245, 158, 11, 0.55)',
            borderRadius: 4,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: textColor, font: { family: 'Inter', size: 11 } }
          }
        },
        scales: {
          x: {
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            grid: { color: gridColor },
            ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } }
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: { color: '#f59e0b', font: { family: 'JetBrains Mono', size: 10 } }
          }
        }
      }
    });
  }

  renderRadarChart(nodeRecord, avgMetrics, isDark = true) {
    const canvas = document.getElementById('radarChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    if (this.radarChart) this.radarChart.destroy();

    // Normalized scores (0-100 scale) for intuitive radar comparison
    const maxDeg = Math.max(...AppState.rawPredictions.map(p => p.degree || 1));
    const maxPR = Math.max(...AppState.rawPredictions.map(p => p.pagerank || 0.001));
    const maxBet = Math.max(...AppState.rawPredictions.map(p => p.betweenness || 0.001));
    const maxCore = Math.max(...AppState.rawPredictions.map(p => p.core_number || 1));

    const nodeVals = [
      Math.min(100, ((nodeRecord.degree || 0) / maxDeg) * 100),
      Math.min(100, ((nodeRecord.in_degree || 0) / maxDeg) * 100),
      Math.min(100, ((nodeRecord.pagerank || 0) / maxPR) * 100),
      Math.min(100, ((nodeRecord.betweenness || 0) / maxBet) * 100),
      Math.min(100, ((nodeRecord.core_number || 0) / maxCore) * 100),
      Math.min(100, ((nodeRecord.closeness || 0.5)) * 100)
    ];

    const avgVals = [
      Math.min(100, (avgMetrics.degree / maxDeg) * 100),
      Math.min(100, (avgMetrics.in_degree / maxDeg) * 100),
      Math.min(100, (avgMetrics.pagerank / maxPR) * 100),
      Math.min(100, (avgMetrics.betweenness / maxBet) * 100),
      Math.min(100, (avgMetrics.core / maxCore) * 100),
      Math.min(100, 50)
    ];

    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';

    this.radarChart = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Degree', 'In-Degree', 'PageRank', 'Betweenness', 'Core No.', 'Closeness'],
        datasets: [
          {
            label: `Node ${nodeRecord.node_id}`,
            data: nodeVals,
            backgroundColor: 'rgba(99, 102, 241, 0.3)',
            borderColor: '#6366f1',
            pointBackgroundColor: '#818cf8',
            pointBorderColor: '#fff',
            pointHoverBackgroundColor: '#fff',
            pointHoverBorderColor: '#6366f1'
          },
          {
            label: 'Network Average',
            data: avgVals,
            backgroundColor: 'rgba(148, 163, 184, 0.15)',
            borderColor: '#94a3b8',
            borderDash: [4, 4],
            pointBackgroundColor: '#94a3b8'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: textColor, font: { family: 'Inter', size: 10 } }
          }
        },
        scales: {
          r: {
            angleLines: { color: gridColor },
            grid: { color: gridColor },
            pointLabels: { color: textColor, font: { family: 'Outfit', size: 10 } },
            ticks: { display: false }
          }
        }
      }
    });
  }

  updateThemeColors(theme) {
    const isDark = theme === 'dark';
    if (this.barChart) {
      const topNodes = AppState.rawPredictions.slice(0, 10);
      this.renderBarChart(topNodes, isDark);
    }
    if (this.diffusionChart && AppState.diffusionData) {
      this.renderDiffusionCurve(AppState.diffusionData.waves, isDark);
    }
  }
}

// ============================================================================
// 9. Main Application Controller
// ============================================================================
class MainApp {
  constructor() {
    this.themeManager = new ThemeManager();
    this.networkGraph = new NetworkGraphController('networkCanvas');
    this.chartManager = new ChartManager();
    this.diffusionPlayer = new DiffusionPlayer(this.networkGraph, this.chartManager);
    this.rankingsTable = new RankingsTableController();

    this.init();
  }

  async init() {
    this.bindEvents();
    await this.checkBackendHealth();
    await this.loadSampleCatalog();

    // Auto-trigger Stanford Twitter graph analysis on first load
    this.executeEgoAnalysis('Dataset/Ego_Twitter Stanford University/twitter/twitter');
  }

  bindEvents() {
    // Quick Run Hero Button
    document.getElementById('quickRunBtn')?.addEventListener('click', () => {
      this.executeEgoAnalysis('Dataset/Ego_Twitter Stanford University/twitter/twitter');
    });

    // Custom Path Input Form
    document.getElementById('pathForm')?.addEventListener('submit', (e) => {
      e.preventDefault();
      const path = document.getElementById('customPathInput').value.trim();
      if (!path) return;
      if (path.endsWith('.csv')) {
        this.executeCsvAnalysis(path, path.split('/').pop());
      } else {
        this.executeEgoAnalysis(path);
      }
    });

    // Drag and Drop File Upload
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');

    if (dropZone && fileInput) {
      dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
      });
      dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
      dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
          this.handleFileUpload(e.dataTransfer.files[0]);
        }
      });
      dropZone.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON' && !e.target.closest('button')) {
          fileInput.click();
        }
      });
      fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          this.handleFileUpload(e.target.files[0]);
        }
      });
    }

    // Export Handlers
    document.getElementById('exportRankingsBtn')?.addEventListener('click', () => this.exportCsv());
    document.getElementById('exportJsonBtn')?.addEventListener('click', () => this.exportJson());

    // Jump to diffusion button
    document.getElementById('openDiffusionBtn')?.addEventListener('click', () => {
      document.getElementById('diffusionSection')?.scrollIntoView({ behavior: 'smooth' });
    });
  }

  async checkBackendHealth() {
    const badge = document.getElementById('serverStatusBadge');
    try {
      const data = await ApiClient.checkHealth();
      if (badge) {
        badge.innerHTML = `
          <span class="pulse-dot"></span>
          <span>Online • ${data.device.toUpperCase()} • PyTorch ${data.torch_version}</span>
        `;
      }
    } catch {
      if (badge) {
        badge.innerHTML = `
          <span class="pulse-dot bg-danger"></span>
          <span class="text-danger">Backend Disconnected</span>
        `;
      }
    }
  }

  async loadSampleCatalog() {
    const container = document.getElementById('presetsContainer');
    if (!container) return;

    try {
      const data = await ApiClient.getSamples();
      if (data.samples && data.samples.length > 0) {
        let html = '';
        data.samples.forEach(s => {
          let badgeBg = 'var(--accent-indigo-subtle)';
          let badgeColor = 'var(--accent-indigo)';
          let icon = 'fa-file-csv';

          if (s.type.includes('Edge List')) {
            badgeBg = 'var(--accent-cyan-subtle)';
            badgeColor = 'var(--accent-cyan)';
            icon = 'fa-project-diagram';
          } else if (s.type.includes('GNN')) {
            badgeBg = 'var(--accent-amber-subtle)';
            badgeColor = 'var(--accent-amber)';
            icon = 'fa-brain';
          } else {
            badgeBg = 'var(--accent-emerald-subtle)';
            badgeColor = 'var(--accent-emerald)';
            icon = 'fa-users';
          }

          html += `
            <div class="col-md-6 col-lg-4">
              <div class="preset-card" onclick="window.app.selectPreset('${s.filepath}', '${s.name}')">
                <div>
                  <div class="d-flex align-items-center justify-content-between mb-2">
                    <span class="preset-badge" style="background: ${badgeBg}; color: ${badgeColor};">
                      <i class="fas ${icon} me-1"></i> ${s.type.split(' ')[0]}
                    </span>
                    <small class="text-dim font-mono">${s.rows} items</small>
                  </div>
                  <h6 class="fw-bold mb-1">${s.name}</h6>
                  <p class="text-muted small mb-3" style="font-size: 0.82rem; line-height: 1.4;">
                    ${s.description}
                  </p>
                </div>
                <div class="d-flex align-items-center justify-content-between border-top pt-2 mt-auto" style="border-color: var(--border-color) !important;">
                  <span class="small fw-semibold" style="color: var(--accent-indigo);">
                    <i class="fas fa-bolt me-1"></i> Analyze Dataset
                  </span>
                  <i class="fas fa-arrow-right text-dim small"></i>
                </div>
              </div>
            </div>
          `;
        });
        container.innerHTML = html;
      }
    } catch (err) {
      container.innerHTML = `<div class="col-12 text-danger text-center py-3">Failed to load samples: ${err.message}</div>`;
    }
  }

  selectPreset(filepath, name) {
    const input = document.getElementById('customPathInput');
    if (input) input.value = filepath;
    this.executeCsvAnalysis(filepath, name);
  }

  async executeEgoAnalysis(subdir) {
    this.showLoading('Executing TA-GAT Graph Intelligence Pipeline...', 'Extracting directed ego topology, normalizing 1,180-dim feature matrix & computing 4-head GAT layers...');
    try {
      const data = await ApiClient.predictGraph(subdir);
      this.handleAnalysisResults(data, 'Stanford Twitter Ego Network');
      ToastManager.show('Successfully executed TA-GAT neural inference on Stanford Twitter graph!', 'success');
    } catch (err) {
      ToastManager.show(`Ego analysis error: ${err.message}`, 'error');
    } finally {
      this.hideLoading();
    }
  }

  async executeCsvAnalysis(path, displayName) {
    this.showLoading(`Loading ${displayName}...`, 'Parsing node features, calculating topological centralities & ranking influencer nodes...');
    try {
      const data = await ApiClient.predictCsv(path);
      this.handleAnalysisResults(data, displayName);
      ToastManager.show(`Loaded and analyzed: ${displayName}`, 'success');
    } catch (err) {
      ToastManager.show(`Analysis failed: ${err.message}`, 'error');
    } finally {
      this.hideLoading();
    }
  }

  async handleFileUpload(file) {
    this.showLoading(`Processing ${file.name}...`, 'Uploading dataset and executing neural influence pipeline...');
    try {
      const data = await ApiClient.uploadDataset(file);
      this.handleAnalysisResults(data, file.name);
      ToastManager.show(`Successfully uploaded and analyzed: ${file.name}`, 'success');
    } catch (err) {
      ToastManager.show(`Upload failed: ${err.message}`, 'error');
    } finally {
      this.hideLoading();
    }
  }

  handleAnalysisResults(data, datasetTitle) {
    AppState.activeDatasetName = datasetTitle;
    AppState.rawPredictions = data.predictions || [];
    AppState.visualizationPayload = data.visualization || null;
    const summary = data.summary || {};

    // Reveal Dashboard
    const dashboard = document.getElementById('resultsDashboard');
    if (dashboard) dashboard.style.display = 'block';

    // Update KPI Cards
    const totalNodes = summary.total_nodes || AppState.rawPredictions.length;
    document.getElementById('kpiNodes').textContent = totalNodes;
    document.getElementById('kpiEdges').textContent = data.visualization ? `${data.visualization.total_graph_edges || 0} Directed Edges` : 'Tabular Dataset';

    const topNode = AppState.rawPredictions[0] || {};
    document.getElementById('kpiTopScore').textContent = topNode.pred ? topNode.pred.toFixed(4) : '0.0000';
    document.getElementById('kpiTopId').textContent = topNode.node_id ? `Node ID: ${topNode.node_id}` : 'ID: ---';

    document.getElementById('kpiAvgScore').textContent = summary.avg_score ? summary.avg_score.toFixed(4) : '0.0000';
    document.getElementById('kpiScoreRange').textContent = `Min: ${summary.min_score ? summary.min_score.toFixed(2) : 0} | Max: ${summary.max_score ? summary.max_score.toFixed(2) : 0}`;

    // Render Vis.js Network
    const isDark = this.themeManager.isDark;
    this.networkGraph.render(data.visualization, isDark);

    // Render Top 10 Influencers Bar Chart
    this.chartManager.renderBarChart(AppState.rawPredictions.slice(0, 10), isDark);

    // Render Paginated Rankings Table
    AppState.page = 1;
    this.rankingsTable.render();

    // Scroll to dashboard smoothly
    dashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  onTableRowClick(nodeId) {
    const record = AppState.rawPredictions.find(p => String(p.node_id) === String(nodeId));
    if (record) {
      this.networkGraph.focusNode(nodeId);
      this.showNodeTelemetry(record);
    }
  }

  showNodeTelemetry(record) {
    AppState.selectedNode = record;

    document.getElementById('modalNodeId').textContent = `Node: ${record.node_id}`;
    document.getElementById('modalTier').textContent = `Rank #${record.rank || '-'}`;
    document.getElementById('modalScore').textContent = (record.pred || 0).toFixed(4);
    document.getElementById('modalDegree').textContent = record.degree !== undefined ? record.degree : '-';
    document.getElementById('modalInDegree').textContent = record.in_degree !== undefined ? record.in_degree : '-';
    document.getElementById('modalOutDegree').textContent = record.out_degree !== undefined ? record.out_degree : '-';
    document.getElementById('modalPageRank').textContent = record.pagerank !== undefined ? parseFloat(record.pagerank).toFixed(5) : '-';
    document.getElementById('modalBetweenness').textContent = record.betweenness !== undefined ? parseFloat(record.betweenness).toFixed(5) : '-';
    document.getElementById('modalCore').textContent = record.core_number !== undefined ? record.core_number : '-';

    // Compute average metrics for comparison
    const len = AppState.rawPredictions.length || 1;
    const avgDegree = AppState.rawPredictions.reduce((acc, p) => acc + (p.degree || 0), 0) / len;
    const avgInDegree = AppState.rawPredictions.reduce((acc, p) => acc + (p.in_degree || 0), 0) / len;
    const avgPR = AppState.rawPredictions.reduce((acc, p) => acc + (p.pagerank || 0), 0) / len;
    const avgBet = AppState.rawPredictions.reduce((acc, p) => acc + (p.betweenness || 0), 0) / len;
    const avgCore = AppState.rawPredictions.reduce((acc, p) => acc + (p.core_number || 0), 0) / len;

    this.chartManager.renderRadarChart(record, {
      degree: avgDegree,
      in_degree: avgInDegree,
      pagerank: avgPR,
      betweenness: avgBet,
      core: avgCore
    }, this.themeManager.isDark);

    const modal = new bootstrap.Modal(document.getElementById('nodeModal'));
    modal.show();
  }

  exportCsv() {
    if (AppState.rawPredictions.length === 0) return;
    const headers = Object.keys(AppState.rawPredictions[0]).join(',');
    const rows = AppState.rawPredictions.map(p => Object.values(p).join(',')).join('\n');
    this.downloadBlob(new Blob([headers + '\n' + rows], { type: 'text/csv' }), 'tagat_influential_nodes.csv');
    ToastManager.show('Exported CSV dataset', 'success');
  }

  exportJson() {
    if (AppState.rawPredictions.length === 0) return;
    const blob = new Blob([JSON.stringify(AppState.rawPredictions, null, 2)], { type: 'application/json' });
    this.downloadBlob(blob, 'tagat_influential_nodes.json');
    ToastManager.show('Exported JSON dataset', 'success');
  }

  downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  showLoading(headline, subline) {
    const el = document.getElementById('loadingIndicator');
    if (!el) return;
    document.getElementById('loadingHeadline').textContent = headline;
    document.getElementById('loadingSubline').textContent = subline;
    el.style.display = 'block';
  }

  hideLoading() {
    const el = document.getElementById('loadingIndicator');
    if (el) el.style.display = 'none';
  }
}

// Global App bootstrap
document.addEventListener('DOMContentLoaded', () => {
  window.app = new MainApp();
});
