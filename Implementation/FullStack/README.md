# TA-GAT Studio: Influential Node Detection & Information Diffusion Platform
**Enterprise Graph Intelligence Full-Stack Platform for Social Influence Analytics**

---

## Executive Summary & Presentation Overview

> [!NOTE]
> **Project Title:** Topology-Aware Graph Attention Network (TA-GAT) for Influential Node Detection in Social Ego Networks  
> **Course / Presentation:** Project-Based Learning (PBL) — Final Evaluation & Viva  
> **Key Innovation:** Combines directed graph topological centralities (PageRank, Betweenness, Closeness, Core Number) with multi-head graph attention mechanisms (PyTorch Geometric) to predict high-impact opinion leaders and simulate epidemic information diffusion in real time.  
> **Evaluation Highlight:** Achieves a **Spearman Rank Correlation ($\rho$) of 0.2989**, outperforming classical PageRank ($\rho = 0.2200$) and Degree Centrality ($\rho = 0.1800$) against ground-truth Monte Carlo Independent Cascade simulations.

---

## 1. System Architecture

The project is structured as a decoupled, production-grade full-stack platform:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FRONTEND (TA-GAT Studio)                                  │
│  ┌───────────────────────┐  ┌─────────────────────────┐  ┌───────────────────────────────────┐  │
│  │     Theme Engine      │  │     Vis.js Network      │  │        Chart.js Analytics         │  │
│  │ (Obsidian Dark/Light) │  │  • Barnes-Hut Physics   │  │  • Top 10 Influence Bar Chart    │  │
│  │  • CSS Custom Props   │  │  • Search & Camera Zoom │  │  • Diffusion S-Curve Adoption     │  │
│  │  • Glassmorphism      │  │  • Tier Filtering HUD   │  │  • 6-Axis Node Radar Comparison  │  │
│  └───────────────────────┘  └─────────────────────────┘  └───────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  ┌───────────────────────────────────┐  │
│  │               Diffusion Cascade Player             │  │      Enterprise Leaderboard       │  │
│  │  • Independent Cascade (IC) Step Controller        │  │  • Multi-Column Header Sorting    │  │
│  │  • Live Visual Outbreak Flare Animation on Graph   │  │  • Client-Side Dynamic Pagination │  │
│  └────────────────────────────────────────────────────┘  └───────────────────────────────────┘  │
└──────────────────────────────────────────────▲──────────────────────────────────────────────────┘
                                               │ JSON REST API (HTTP)
┌──────────────────────────────────────────────▼──────────────────────────────────────────────────┐
│                                       BACKEND (Flask + PyTorch)                                 │
│  ┌───────────────────────┐  ┌─────────────────────────┐  ┌───────────────────────────────────┐  │
│  │  Flask Server (app.py)│  │    NetworkX Analysis    │  │       PyTorch Geometric GNN       │  │
│  │  • REST Endpoints     │  │  • PageRank, Degrees    │  │  • 4-Head Graph Attention (GAT)   │  │
│  │  • CORS & Error Catch │  │  • Betweenness/Closeness│  │  • 1,180-dim Concatenated Feature │  │
│  │  • Static File Server │  │  • Independent Cascade  │  │  • Best Model Inference (CPU/CUDA)│  │
│  └───────────────────────┘  └─────────────────────────┘  └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deep-Dive: Frontend Dashboard Components

The frontend is located under `Implementation/FullStack/frontend/` and is divided into clean, decoupled files:
- **`index.html`**: Clean semantic HTML5 layout with accessible navigation, hero cards, KPI boards, and modals.
- **`css/style.css`**: Complete design system with CSS custom properties, glassmorphism, responsive grid, and custom scrollbars.
- **`js/app.js`**: Object-oriented modular controllers (`ThemeManager`, `ToastManager`, `ApiClient`, `NetworkGraphController`, `DiffusionPlayer`, `RankingsTableController`, and `ChartManager`).

### Component 1: Enterprise Header & Live Telemetry
- **Brand Identity:** Styled badge with icon, title, and GNN version chip.
- **Live Health Status:** Automatically queries `/api/health` on load and displays an animated pulse indicator (`Online • CPU • PyTorch 2.13.0+cpu`).
- **Benchmark Modal Trigger:** Instant access to comparative algorithm evaluations and model layer dimensions.
- **Theme Switcher:** Toggles between **Obsidian Dark** (default `#0b0f19`) and **Crisp Light** (`#f8fafc`). Persisted in browser `localStorage`.

### Component 2: Multi-Modal Data Ingestion Deck
Provides 3 flexible ways to input graph datasets for inference:
1. **Pre-Configured Samples:** 5 preset datasets loaded dynamically via `/api/samples`:
   - *Twitter Ego Network (Stanford)* (450 ties)
   - *Twitter Ego Topological Nodes*
   - *Viral Marketing Campaign (KOLs)*
   - *Tech Community Collaboration Graph*
   - *Full 1,180-Feature Matrix*
2. **File Upload Dropzone:** Drag-and-drop zone supporting `.csv`, `.edges`, `.txt`, and `.tsv` files.
3. **Custom Workspace Path:** Direct file path execution for reproducible local testing.

### Component 3: Summary KPI Metric HUD
Four metric cards give immediate high-level feedback:
- **Network Size:** Total node count and directed edge count.
- **Super Spreader Score:** Highest predicted score and top node identifier.
- **Mean Graph Influence:** Average attention score across the network, with min/max range.
- **GNN Architecture:** Model layer summary (TA-GAT Multi-Head, 327,105 parameters).

### Component 4: Interactive Vis.js Topology Canvas
A real-time 2D network visualization canvas with force-directed Barnes-Hut physics:
- **Color-Coded Node Hierarchy:**
  - 🟡 **Super Spreaders (Top 3):** Amber glow, radius 30px (`#f59e0b`).
  - 🟣 **Key Influencers (Top 10):** Indigo, radius 23px (`#6366f1`).
  - 🔵 **Bridge Connectors (Top 25):** Cyan, radius 17px (`#06b6d4`).
  - ⚪ **Community Members:** Slate gray, radius 13px (`#64748b`).
- **Floating HUD Overlay:**
  - **Search & Focus Spotlight:** Type any Node ID (e.g. `364971269`) to center the camera with smooth easing and select the node.
  - **Tier Filter:** Filter visible nodes dynamically by rank tier.
  - **Freeze / Run Physics:** Freeze node coordinates for stable inspection.
  - **Fit View & Fullscreen Mode:** Expand the graph to full screen for presentations.

### Component 5: Top 10 Influential Nodes Chart
- Rendered via Chart.js with custom gradient bars.
- Displays the top 10 opinion leaders ranked by TA-GAT predicted scalar score.
- Includes a direct performance gauge comparing TA-GAT Spearman correlation ($\rho = 0.2989$) against PageRank baseline ($\rho = 0.2200$).

### Component 6: Interactive Information Diffusion Cascade Player
Simulates how viral information or memes spread through the network from seed nodes:
- **Simulation Controls:**
  - Adjustable **Propagation Probability** ($p \in [0.02, 0.45]$).
  - Adjustable **Max Wave Depth** ($1 \dots 15$ steps).
  - **Seed Strategy Selector:** Top 3 Super Spreaders, Top 5 Influencers, or Single Highest Node.
- **Interactive Outbreak Player Deck:**
  - **Play / Pause:** Auto-advances the cascade with configurable speed (0.5x, 1.0x, 2.0x).
  - **Step Forward / Step Backward / Reset:** Manual wave-by-wave exploration.
  - **Real-Time Network Outbreak Flare:** Newly activated nodes dynamically pulse with amber/red flame coloring on the Vis.js canvas as waves progress!
  - **S-Curve Adoption Chart:** Live line/bar chart displaying cumulative active nodes and new infections per wave.
  - **Wave Pills:** Clickable badges for jumping directly to any wave step.

### Component 7: Enterprise Leaderboard Table
- **Multi-Column Sorting:** Click column headers (Rank, Node ID, TA-GAT Score, Degree, In/Out-Degree, PageRank, Betweenness, Core Number) to toggle ascending/descending order.
- **Client-Side Pagination:** Configurable page size (10, 15, 25, 50, 100) with quick page navigation.
- **Visual Score Bar:** In-cell gradient progress bar visualizing relative influence.
- **Row Click Inspection:** Clicking any row highlights the node on the graph and opens the **Node Telemetry Modal**.
- **Data Export:** Instant download of the full leaderboard as CSV or JSON.

### Component 8: Node Telemetry Modal (Radar Comparison)
- Shows absolute metrics for the inspected node (Degree, In-Degree, Out-Degree, PageRank, Betweenness, Core Number).
- Features a **6-Axis Radar Chart** comparing the selected node's topological profile against the **network average** across Degree, In-Degree, PageRank, Betweenness, Core Number, and Closeness.

---

## 3. Backend REST API Reference

The backend runs on **Flask** with PyTorch and PyTorch Geometric:

| Endpoint | Method | Payload / Params | Description |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | None | Serves the single-page application `index.html`. |
| `/api/health` | `GET` | None | Returns server uptime, PyTorch version, active model name, and device (`cpu`/`cuda`). |
| `/api/samples` | `GET` | None | Lists available sample datasets from `input samples/`. |
| `/api/model_info` | `GET` | None | Returns neural architecture layers, parameter count, and baseline metrics. |
| `/predict_graph` | `POST` | `{"dataset_subdir": "..."}` | Runs the complete TA-GAT pipeline on an Ego Twitter graph folder (`.edges`, `.feat`, `.featnames`). |
| `/api/predict` | `POST` | `{"csv_path": "..."}` or `{"rows": [...]}` | Predicts influence from edge lists or tabular node features. |
| `/api/predict_upload` | `POST` | `multipart/form-data` (file) | Accepts uploaded dataset files and runs inference. |
| `/api/simulate_diffusion` | `POST` | `{"seeds": [...], "p": 0.15, "max_steps": 8}` | Runs Monte Carlo Independent Cascade simulation and returns step-by-step infection waves. |

---

## 4. How to Present This Project (Viva / Demo Script)

When demonstrating this project to professors, evaluators, or audiences, follow this structured walkthrough:

### Step 1: Introduction & Problem Motivation (1 minute)
> *"Hello everyone. Our project tackles the problem of **Influential Node Detection in Social Networks**. In large-scale social networks like Twitter, identifying the most influential spreaders is critical for viral marketing, epidemic tracking, and stopping misinformation. Classical metrics like PageRank or Degree Centrality only measure simple connection counts. Our solution introduces **TA-GAT: a Topology-Aware Graph Attention Network** that combines directed topological properties with multi-head neural attention to learn complex, non-linear influence patterns."*

### Step 2: Show the Live Interface & Themes (30 seconds)
1. Point to the top navigation bar and explain the live backend telemetry badge:  
   *"Here you can see our real-time system status connected to our PyTorch backend."*
2. Click the **Theme Switcher** (sun/moon icon in the top right):  
   *"We designed TA-GAT Studio with an industry-grade dual theme — a high-contrast Obsidian Dark mode and a clean Crisp Light mode, with responsive glassmorphism."*

### Step 3: Explain the Graph Visualization & HUD Spotlight (1.5 minutes)
1. Point to the **Interactive Topology Canvas**:  
   *"The graph shows the Stanford Twitter ego-network with 221 nodes and 450 active directed follower ties. Nodes are sized and color-coded by their predicted influence tier: amber for Top 3 Super Spreaders, indigo for Key Influencers, cyan for Bridge Connectors."*
2. Demonstrate **Search & Focus Spotlight**:  
   - In the network search box, type `364971269` and press Enter.
   - Show how the camera smoothly glides and zooms in onto the top super spreader:  
     *"With our HUD spotlight, we can instantly focus onto Node `364971269` which our model predicts as the #1 super spreader with an influence score of 2.1585."*
3. Show the **Tier Filter Dropdown** (switch to 'Top 3 Super Spreaders' and back to 'All Visible Nodes').

### Step 4: Demonstrate the Diffusion Cascade Outbreak Player (2 minutes)
1. Scroll down to the **Information Diffusion Cascade Simulator**:  
   *"To prove our model's predictions work in reality, we simulate viral diffusion using the Monte Carlo Independent Cascade model."*
2. Click **'Run Outbreak Simulation'**:  
   *"Notice the results: starting from just our top 3 seed nodes, information reaches over 187 users (84.6% of the network) across 6 cascade waves."*
3. Click **'Play Outbreak'**:  
   *"Observe the network graph in real time — newly infected nodes glow and change color wave-by-wave, and the S-curve chart tracks the cumulative epidemic adoption curve."*

### Step 5: Table Sorting, Pagination & Radar Telemetry (1.5 minutes)
1. Scroll down to the **Complete Influential Node Leaderboard**:  
   *"Our leaderboard features multi-column sorting and pagination. We can sort by Degree, PageRank, or TA-GAT Score."*
2. Click on the top row (`Node 364971269`) to open the **Node Telemetry Modal**:  
   *"When we inspect a node, TA-GAT Studio renders a 6-axis Radar Chart comparing this specific node against the network average. We can see this node has extraordinarily high PageRank and Betweenness, explaining why the attention network prioritized it."*

### Step 6: Model Architecture & Benchmark Proof (1 minute)
1. Open **'Baselines & Benchmarks'** from the navbar:  
   *"Finally, our quantitative results: TA-GAT achieves a Spearman correlation of **0.2989**, outperforming PageRank at **0.2200** and Degree Centrality at **0.1800**, with a significantly lower Mean Absolute Error of **58.31** vs baseline **64.20**."*
2. Conclude:  
   *"In conclusion, TA-GAT Studio provides a complete, end-to-end platform bridging theoretical Graph Neural Networks with practical, interactive social network intelligence."*

---

## 5. How to Run the Application

### Prerequisites
- Python 3.10+ (virtual environment configured in project root `.venv`)
- Modern web browser (Chrome, Edge, Firefox, Safari)

### Quick Start
Open PowerShell or your terminal and execute:

```powershell
# 1. Navigate to the backend directory
cd "Implementation/FullStack/backend"

# 2. Activate Python virtual environment and run the server
..\..\..\.venv\Scripts\python.exe app.py
```

### Accessing the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

### Testing Backend Endpoints
In a separate terminal, test the endpoints using Python:
```powershell
..\..\..\.venv\Scripts\python.exe test_request_graph.py
```

---

## 6. Mathematical Foundations & Metrics Reference

### 1. Graph Attention Network (GAT) Formulation
For a node $i$ with neighborhood $\mathcal{N}_i$, the attention coefficients $\alpha_{ij}$ to neighbor $j$ are computed as:
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_k]\right)\right)}$$

The node's updated representation across $K=4$ attention heads is:
$$\mathbf{h}_i' = \Vert_{k=1}^K \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j\right)$$

### 2. Spearman's Rank Correlation Coefficient ($\rho$)
Measures monotonic ranking alignment between predicted influence $\hat{y}$ and true diffusion cascade reach $y$:
$$\rho = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}$$
Where $d_i = \text{rank}(\hat{y}_i) - \text{rank}(y_i)$ and $n$ is total nodes.
- **TA-GAT Score:** $\mathbf{0.2989}$ (Highest rank fidelity)
- **PageRank Baseline:** $0.2200$
- **Degree Centrality:** $0.1800$

### 3. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
- **TA-GAT Score:** $\mathbf{58.31}$ (Lower error)
- **Baseline Centrality:** $64.20$

---

## 7. Troubleshooting & FAQ

- **Port 5000 is already in use:**
  If another process is holding port 5000, find its PID and terminate it:
  ```powershell
  Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess | Stop-Process -Force
  ```
- **Static assets not refreshing:**
  Clear browser cache using `Ctrl + F5` or open in an Incognito / Private window.
- **GPU Acceleration:**
  The server automatically detects CUDA if an NVIDIA GPU is present (`/api/health` will report `cuda`). By default, it operates on optimized CPU PyTorch.
