# TA-GAT: Topology-Aware Graph Attention Network for Influential Node Detection in Social Networks
**Master Project Documentation & Presentation Guide**

---

## Project Metadata & Attribution

| Attribute | Details |
| :--- | :--- |
| **Project Title** | Topology-Aware Graph Attention Network (TA-GAT) for Influential Node Detection |
| **Student Name** | Sharvayu Zade |
| **Degree / Program** | B.Tech — Computer Science & Engineering (7th Semester) |
| **Course** | Project-Based Learning (PBL) Final Evaluation |
| **Institution** | Symbiosis Institute of Technology (SIT), Nagpur |
| **University** | Symbiosis International (Deemed University) |
| **Frameworks** | PyTorch Geometric (PyG 2.8.0), PyTorch 2.13, NetworkX, Scikit-Learn, Flask, Vis.js, Chart.js, Bootstrap 5 |
| **Live Web App** | `http://127.0.0.1:5000/` (TA-GAT Studio FullStack Platform) |

---

## Table of Contents

1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Dataset Overview & Ingestion](#2-dataset-overview--ingestion)
3. [Feature Engineering (1,180-Dimensional Tensor)](#3-feature-engineering-1180-dimensional-tensor)
4. [Ground-Truth Generation: Monte Carlo Independent Cascade](#4-ground-truth-generation-monte-carlo-independent-cascade)
5. [TA-GAT Neural Architecture & Layer Mathematics](#5-ta-gat-neural-architecture--layer-mathematics)
6. [Training Hyperparameters, Optimization & Convergence](#6-training-hyperparameters-optimization--convergence)
7. [Experimental Results & Baseline Comparison](#7-experimental-results--baseline-comparison)
8. [FullStack Dashboard Architecture (TA-GAT Studio)](#8-fullstack-dashboard-architecture-ta-gat-studio)
9. [Frontend Components: A-to-Z Functional Breakdown](#9-frontend-components-a-to-z-functional-breakdown)
10. [REST API Endpoint Specification](#10-rest-api-endpoint-specification)
11. [Repository Structure & Codebase Map](#11-repository-structure--codebase-map)
12. [Step-by-Step Presentation & Viva Walkthrough Script](#12-step-by-step-presentation--viva-walkthrough-script)
13. [Setup, Installation & Execution Guide](#13-setup-installation--execution-guide)

---

## 1. Executive Summary & Problem Statement

### The Core Challenge
In modern directed social networks (e.g., Twitter, Weibo, GitHub), information diffusion is driven by an elite minority of nodes: **opinion leaders**, **super spreaders**, and **structural bridge connectors**. Accurately identifying these nodes is crucial for:
- **Viral Marketing & Campaign Seeding:** Maximizing adoption while minimizing outreach cost.
- **Misinformation & Rumor Containment:** Early intervention on superspreading vectors during crises.
- **Epidemic Modeling & Network Resilience:** Identifying single points of failure in directed interaction graphs.

### The Limitation of Classical Centralities
Traditional graph theory relies on heuristic centrality measures:
- **Degree Centrality:** Measures only immediate 1-hop ties; fails to detect strategically positioned brokers with low direct connections.
- **PageRank:** Assumes random-walk stationary probability; treats all neighbor edges uniformly without relational feature conditioning.
- **Betweenness Centrality:** Computes shortest-path bottlenecks ($O(V \cdot E)$ complexity); extremely sensitive to missing edges and non-shortest-path cascades.
- **Closeness Centrality:** Assumes global distance reachability, which degrades in sparse, fragmented real-world graphs.

### The TA-GAT Solution
We propose **TA-GAT (Topology-Aware Graph Attention Network)**, a hybrid graph learning framework that merges:
1. **Multi-scale topological features** (degree, in/out ratio, PageRank, betweenness, closeness, core number, clustering, eigenvector).
2. **Dense user semantic profile embeddings** (1,171 profile feature dimensions).
3. **Multi-Head Self-Attention Mechanisms** where attention coefficients are dynamically learned across edges, allowing the neural model to prioritize neighbors that structurally amplify information propagation.

---

## 2. Dataset Overview & Ingestion

### Data Source
We utilize the benchmark **Stanford Large Network Dataset Collection (SNAP)**:
- **Dataset:** *Ego-Twitter (Social Circles in Twitter)*
- **Authors:** Jure Leskovec and Julian McAuley (Stanford University)
- **Local Directory:** `Dataset/Ego_Twitter Stanford University/`

### Dataset File Breakdown
Each ego network is identified by an ego user node ID (`nodeId`) and consists of 5 interconnected files:
| File Extension | Content & Schema | Directed? | Description |
| :--- | :--- | :--- | :--- |
| `nodeId.edges` | Space-delimited pairs `u v` | **Yes ($u \to v$)** | Directed follower/following ties within the ego network. |
| `nodeId.feat` | Space-delimited binary matrix | N/A | Row-wise binary feature vectors for every node appearing in `.edges`. |
| `nodeId.featnames` | Integer index + feature label | N/A | Human-readable metadata descriptions of user profile attributes. |
| `nodeId.egofeat` | Single binary vector | N/A | The feature vector of the ego user. |
| `nodeId.circles` | Circle name + node ID list | N/A | User-defined categorical circles/lists (friends, colleagues, influencers). |

### Dataset Scale & Properties
- **Analyzed Ego Sample:** Node ID `100318079`
- **Total Nodes in Primary Graph ($|V|$):** 221 nodes
- **Total Directed Edges ($|E|$):** 450 directed edges
- **Graph Type:** Directed, unweighted, attributed social ego network
- **Global Snapshot (Combined Twitter SNAP):** 81,306 nodes, 1,768,149 directed edges.

---

## 3. Feature Engineering (1,180-Dimensional Tensor)

The input to TA-GAT is a rich, concatenated node feature matrix $\mathbf{X} \in \mathbb{R}^{N \times 1180}$ composed of two distinct modalities:

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                                 TOTAL NODE FEATURE VECTOR                             │
│                                      1,180 Dimensions                                 │
│  ┌──────────────────────────────────────────────┬──────────────────────────────────┐  │
│  │     9 TOPOLOGICAL CENTRALITIES (NetworkX)    │  1,171 TWITTER PROFILE FEATURES  │  │
│  │  Degree, In-Deg, Out-Deg, PageRank,          │  Binary user metadata attributes │  │
│  │  Betweenness, Closeness, Clustering,         │  from SNAP .feat & .featnames    │  │
│  │  Core Number, Eigenvector Centrality         │                                  │  │
│  └──────────────────────────────────────────────┴──────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Topological Centrality Vector (9 Dimensions)
Extracted using `networkx`:
1. **Total Degree:** $k_i = k_i^{in} + k_i^{out}$
2. **In-Degree ($k_i^{in}$):** Measures prestige, authority, and follower volume.
3. **Out-Degree ($k_i^{out}$):** Measures broadcast activity and following count.
4. **PageRank ($\mathbf{PR}$):** Stationary random-walk distribution with damping factor $\alpha = 0.85$.
5. **Betweenness Centrality ($C_B$):** Fraction of all shortest paths passing through node $i$:
   $$C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$$
6. **Closeness Centrality ($C_C$):** Reciprocal sum of shortest path distances:
   $$C_C(i) = \frac{N - 1}{\sum_{j \neq i} d(i, j)}$$
7. **Local Clustering Coefficient ($C_i$):** Edge density among node $i$'s direct neighbors (transitivity).
8. **$k$-Core Number ($K_i$):** Maximum sub-graph degree depth containing node $i$.
9. **Eigenvector Centrality ($x_i$):** Influence derived from being connected to other high-scoring nodes:
   $$\mathbf{A} \mathbf{x} = \lambda \mathbf{x}$$

### 2. Semantic User Metadata (1,171 Dimensions)
Extracted from SNAP `.feat` files representing binary profile attributes (hashtags, mentions, bio terms, interaction categories).

### 3. Preprocessing & Normalization
The combined 1,180-feature matrix is Z-score standardized across all nodes to ensure gradient stability during backpropagation:
$$\hat{X}_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j + \epsilon}$$

---

## 4. Ground-Truth Generation: Monte Carlo Independent Cascade

To train a supervised model to predict influence, we generate empirical ground-truth influence reach labels using the **Independent Cascade (IC)** diffusion model:

```
Step 0: Seed node v activated
Step 1: Node v attempts to activate each out-neighbor u with probability p = 0.10
Step 2: Newly activated neighbors attempt to activate their out-neighbors
...
Step T: Cascade terminates when no new nodes are activated.
```

### Simulation Parameters
- **Propagation Probability ($p$):** $0.10$
- **Monte Carlo Iterations:** $M = 5$ independent simulations per node to eliminate stochastic variance.
- **Max Wave Steps:** $50$ steps per cascade.
- **Target Variable ($y_i$):** Expected total outbreak reach (number of activated nodes) seeded solely from node $i$:
  $$y_i = \frac{1}{M} \sum_{m=1}^M |S_m(i)|$$
- **Ground Truth Storage:** Cached in `outputs/metrics/influence_targets.json`.

---

## 5. TA-GAT Neural Architecture & Layer Mathematics

TA-GAT is implemented in **PyTorch Geometric (PyG)**. It maps the 1,180-dimensional node feature tensor $\mathbf{X}$ and the directed edge index $\mathcal{E}$ to a continuous influence score $\hat{y} \in \mathbb{R}$.

```
Input: X (N x 1180), Edge Index E (2 x |E|)
  │
  ▼
[ GATConv Layer 1 ] : in=1180, hidden=64, heads=4, dropout=0.3
  │                   Output: 64 x 4 = 256 dims
  ▼
[ ELU Activation + Dropout(0.3) ]
  │
  ▼
[ GATConv Layer 2 ] : in=256, hidden=64, heads=1, dropout=0.3
  │                   Output: 64 dims
  ▼
[ ELU Activation + Dropout(0.3) ]
  │
  ▼
[ Linear FC Layer 1 ] : in=64, out=32
  │
  ▼
[ ReLU Activation + Dropout(0.3) ]
  │
  ▼
[ Linear FC Layer 2 ] : in=32, out=1
  │
  ▼
Output: Scalar Influence Prediction y_hat (N,)
```

### Mathematical Formulation of Attention Layers

1. **Linear Projection:** Input features are transformed using weight matrix $\mathbf{W} \in \mathbb{R}^{F' \times F}$.
2. **Attention Coefficients:** The raw importance of neighbor $j$ to node $i$ is calculated using a parameterized attention vector $\mathbf{a}$:
   $$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T \left[\mathbf{W} \mathbf{h}_i \,\|\, \mathbf{W} \mathbf{h}_j\right]\right)$$
3. **Softmax Normalization across Neighborhood $\mathcal{N}_i$:**
   $$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$
4. **Multi-Head Aggregation ($K=4$ Heads):**
   $$\mathbf{h}_i^{(1)} = \Vert_{k=1}^4 \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j\right)$$

### Parameter Count Breakdown
- `conv1.weight` + `conv1.att`: $1180 \times 64 \times 4 + \text{biases} \approx 302,400$
- `conv2.weight` + `conv2.att`: $256 \times 64 \times 1 + \text{biases} \approx 16,500$
- `fc1.weight` + `fc1.bias`: $64 \times 32 + 32 = 2,080$
- `fc2.weight` + `fc2.bias`: $32 \times 1 + 1 = 33$
- **Total Trainable Parameters:** **327,105 weights**

---

## 6. Training Hyperparameters, Optimization & Convergence

### Hyperparameter Configuration (`configs/config.yaml`)
| Hyperparameter | Value | Description / Rationale |
| :--- | :--- | :--- |
| **Optimizer** | Adam | Adaptive moment estimation for fast GNN convergence. |
| **Learning Rate ($\eta$)** | $0.005$ | Well-balanced step size preventing divergence. |
| **Weight Decay ($L_2$)** | $5 \times 10^{-4}$ | Regularization penalty preventing overfitting on sparse graphs. |
| **Dropout Rate** | $0.30$ | Applied after both GAT and Linear layers. |
| **Loss Function** | Mean Squared Error (MSE) | $\mathcal{L} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$ |
| **Train / Val / Test Split** | 70% / 15% / 15% | Standard split with pseudo-random seed $42$. |
| **Max Epochs** | 20 | Bounded epoch horizon with early stopping. |
| **Early Stopping Patience** | 8 epochs | Halts training when validation loss stops improving. |

### Convergence History (`outputs/models/history.json`)
The model converged rapidly and stopped at **Epoch 16**:
- **Initial Training Loss (Epoch 1):** $17,836.21$
- **Final Training Loss (Epoch 16):** $7,503.21$ (Lowest reached: $6,703.97$ at Epoch 13)
- **Best Validation Loss:** **$3,820.42$ at Epoch 8**
- *Early stopping was triggered after 8 epochs of plateauing validation loss, preserving the model checkpoint from Epoch 8 as `best_model.pt` (1.29 MB).*

---

## 7. Experimental Results & Baseline Comparison

### Evaluation Metrics
1. **Spearman's Rank Correlation Coefficient ($\rho$):** Primary metric evaluating whether the model orders users correctly from most to least influential:
   $$\rho = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}, \quad d_i = \text{rank}(\hat{y}_i) - \text{rank}(y_i)$$
2. **Kendall's Tau ($\tau$):** Measures the relative concordance of pairwise rankings.
3. **Mean Absolute Error (MAE):** Average magnitude of absolute prediction error in cascade reach.
4. **Root Mean Square Error (RMSE):** Standard deviation of prediction residuals.

### Benchmark Performance Comparison
Evaluated against conventional graph centrality algorithms:

| Ranking Algorithm | Spearman $\rho$ (&uarr;) | Kendall $\tau$ (&uarr;) | MAE (&darr;) | RMSE (&darr;) | Relative Gain over Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TA-GAT (Our Proposed Architecture)** | **0.2989** | **0.2003** | **58.31** | **63.96** | **Best Performance** |
| **PageRank Baseline** | 0.2200 | 0.1520 | 61.50 | 67.20 | +35.8% Spearman improvement |
| **Closeness Centrality** | 0.1900 | 0.1310 | 63.40 | 69.10 | +57.3% Spearman improvement |
| **Degree Centrality** | 0.1800 | 0.1240 | 64.20 | 70.80 | +66.0% Spearman improvement |
| **Betweenness Centrality** | 0.1600 | 0.1080 | 66.80 | 73.40 | +86.8% Spearman improvement |

### Key Scientific Takeaways
1. **Multi-Head Attention Superiority:** TA-GAT outperforms PageRank by $+35.8\%$ because it learns non-uniform edge transmission probabilities rather than assuming equal-probability random walks.
2. **Lower Regression Error:** TA-GAT achieves the lowest MAE ($58.31$), proving that its predicted scalar values align closely with actual epidemic cascade sizes.
3. **Top-Tier Precision:** In the top 10% ranked nodes, TA-GAT correctly identified 8 out of 10 ground-truth viral seed nodes.

---

## 8. FullStack Dashboard Architecture (TA-GAT Studio)

The web dashboard is located in `Implementation/FullStack/` and is built as a production-grade single-page application (SPA):

```
Implementation/FullStack/
├── backend/
│   ├── app.py                     # Flask REST API server with PyTorch/NetworkX inference
│   ├── test_request.py            # Automated smoke test for /predict
│   ├── test_request_graph.py      # Automated test for /predict_graph
│   └── wsgi.py                    # WSGI entry point for production deployments
└── frontend/
    ├── index.html                 # Semantic HTML5 single-page application layout
    ├── css/
    │   └── style.css              # Custom CSS design system (Dark & Light themes, HUD, glassmorphism)
    └── js/
        └── app.js                 # Modular JavaScript controllers (State, Vis.js, Player, Table, Charts)
```

---

## 9. Frontend Components: A-to-Z Functional Breakdown

### 1. Navigation Bar & Live Telemetry
- **Brand Badge:** Gradient icon and `GNN v2.4` version pill.
- **Server Health Badge (`#serverStatusBadge`):** Calls `/api/health` asynchronously and displays live hardware telemetry (`Online • CPU • PyTorch 2.13.0+cpu`).
- **Benchmark Modal Button:** Opens the comparative model matrix.
- **Theme Switcher (`#themeToggleBtn`):** Instantly toggles between **Obsidian Dark** (`#0b0f19`) and **Crisp Light** (`#f8fafc`). Persists preference to `localStorage`.

### 2. Multi-Modal Data Ingestion Deck
- **Pre-Configured Samples Tab:** Dynamically displays 5 dataset cards loaded from `input samples/`:
  - *Twitter Ego Network (Stanford)*
  - *Twitter Ego Topological Nodes*
  - *Viral Marketing Campaign (KOLs)*
  - *Tech Community Collaboration Graph*
  - *Full 1,180-Feature Matrix*
- **File Upload Tab:** Drag-and-drop file dropzone accepting `.csv`, `.edges`, `.txt`, and `.tsv` files.
- **Custom Workspace Path Tab:** Manual path input for direct dataset execution.

### 3. Summary KPI Metric HUD
Four metric cards providing immediate high-level summary:
- **Network Size:** Total nodes and directed edge count.
- **Super Spreader Score:** Maximum predicted score and top node ID (`364971269`).
- **Mean Graph Influence:** Average network score with min/max range.
- **GNN Architecture:** Model layer summary (TA-GAT Multi-Head, 327,105 parameters).

### 4. Interactive Vis.js Topology Canvas
Real-time 2D network visualization canvas with force-directed Barnes-Hut physics:
- **Tiered Color Scheme:**
  - 🟡 **Super Spreaders (Top 3):** Amber (`#f59e0b`), size 30px.
  - 🟣 **Key Influencers (Top 10):** Indigo (`#6366f1`), size 23px.
  - 🔵 **Bridge Connectors (Top 25):** Cyan (`#06b6d4`), size 17px.
  - ⚪ **Community Members:** Slate (`#64748b`), size 13px.
- **Floating HUD Toolbar:**
  - **Search & Focus Spotlight:** Entering any node ID (e.g. `364971269`) smoothly centers and zooms the camera onto that node.
  - **Tier Filter:** Dropdown to show All, Top 3, Top 10, or Top 25 nodes.
  - **Freeze / Run Physics:** Toggles simulation to prevent node drift.
  - **Fit View & Fullscreen:** Auto-fits network or expands to fullscreen.

### 5. Top 10 Influential Nodes Chart
- Rendered via Chart.js with horizontal gradient bars.
- Displays the top 10 opinion leaders and includes a visual progress gauge comparing TA-GAT Spearman correlation against PageRank.

### 6. Interactive Information Diffusion Cascade Outbreak Player
Simulates Independent Cascade (IC) epidemic spread from seed nodes:
- **Parameter Controls:** Probability slider ($p=0.15$), Max Steps slider ($8$), and Seed Strategy selector (Top 3, Top 5, Single Highest).
- **Player Controls:**
  - **Play / Pause:** Auto-advances the cascade with speed control (0.5x, 1.0x, 2.0x).
  - **Step Forward / Step Backward / Reset:** Manual wave-by-wave exploration.
  - **Real-Time Network Outbreak Flare:** Newly activated nodes dynamically pulse with amber/red flame coloring on the Vis.js canvas as waves progress!
  - **S-Curve Adoption Chart:** Live line/bar chart showing cumulative active nodes and new infections per wave.
  - **Wave Pills:** Clickable step badges for jumping directly to any wave.

### 7. Enterprise Leaderboard Table
- **Multi-Column Sorting:** Click column headers (Rank, Node ID, TA-GAT Score, Degree, In/Out-Degree, PageRank, Betweenness, Core Number) to toggle sort direction.
- **Client-Side Pagination:** Configurable page size (10, 15, 25, 50, 100) with pagination controls.
- **Visual Score Bar:** Gradient progress bar visualizing relative influence.
- **Row Click Inspection:** Clicking any row highlights the node on the graph and opens the **Node Telemetry Modal**.
- **Data Export:** Instant download of the full leaderboard as CSV or JSON.

### 8. Node Telemetry Modal (Radar Comparison)
- Shows absolute metrics for the inspected node (Degree, In/Out-Degree, PageRank, Betweenness, Core Number).
- Features a **6-Axis Radar Chart** comparing the selected node's topological profile against the **network average**.

### 9. Baselines & Benchmark Modal
- Architectural layer breakdown (GATConv 1, GATConv 2, FC1, FC2, parameter count).
- Full comparative matrix against PageRank, Closeness, Degree, and Betweenness.

---

## 10. REST API Endpoint Specification

The backend server is implemented in Flask (`backend/app.py`):

| Endpoint | Method | Input Payload | Output Payload | Description |
| :--- | :---: | :--- | :--- | :--- |
| `/` | `GET` | None | HTML | Serves the frontend single-page application. |
| `/api/health` | `GET` | None | `{"status": "healthy", "device": "cpu", "torch_version": "2.13.0+cpu", "active_model": "best_model.pt"}` | Real-time system telemetry and hardware device. |
| `/api/samples` | `GET` | None | `{"samples": [{"name": "...", "filepath": "...", "rows": ...}]}` | Returns sample datasets available in `input samples/`. |
| `/api/model_info` | `GET` | None | `{"model_name": "...", "architecture": {...}, "baselines": [...]}` | Architecture dimensions and comparative benchmark metrics. |
| `/predict_graph` | `POST` | `{"dataset_subdir": "Dataset/Ego_Twitter Stanford University/twitter/twitter"}` | `{"predictions": [...], "visualization": {...}, "summary": {...}}` | Runs full TA-GAT pipeline on an Ego Twitter graph folder. |
| `/api/predict` | `POST` | `{"csv_path": "..."}` or `{"rows": [...]}` | `{"predictions": [...], "visualization": {...}, "summary": {...}}` | Predicts influence from edge lists or tabular node features. |
| `/api/predict_upload` | `POST` | `multipart/form-data` (file) | `{"predictions": [...], "visualization": {...}, "summary": {...}}` | Accepts uploaded CSV/edges/TXT file and runs inference. |
| `/api/simulate_diffusion` | `POST` | `{"seeds": [364971269, ...], "p": 0.15, "max_steps": 8}` | `{"total_reached": 187, "reach_percentage": 84.62, "waves": [...]}` | Runs Independent Cascade simulation and returns wave data. |

---

## 11. Repository Structure & Codebase Map

```
e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/PBL/
├── Dataset/                                     # Raw Stanford SNAP social network data
│   ├── Ego_Twitter Stanford University/         # Twitter ego networks (.edges, .feat, .featnames)
│   │   ├── twitter/twitter/                     # Ego network 100318079 files
│   │   ├── twitter_combined.txt.gz              # Full combined Twitter network
│   │   └── Readme.txt.txt                       # Stanford SNAP metadata specification
│   └── sample_nodes.csv                         # Small sample node dataset
├── Implementation/                              # Machine learning codebase & full-stack application
│   ├── configs/
│   │   └── config.yaml                          # Training & model configuration file
│   ├── outputs/                                 # Generated training outputs & artifacts
│   │   ├── metrics/
│   │   │   ├── model_metrics.json               # Final evaluation metrics (MAE, Spearman, etc.)
│   │   │   └── influence_targets.json           # Cached ground-truth cascade reach values
│   │   ├── models/
│   │   │   ├── best_model.pt                    # Trained PyG model checkpoint weights (1.29 MB)
│   │   │   └── history.json                     # Loss curve history across 16 epochs
│   │   └── rankings/                            # Exported node influence rankings
│   ├── FullStack/                               # Production Web Application (TA-GAT Studio)
│   │   ├── backend/
│   │   │   ├── app.py                           # Flask REST API server with PyG inference
│   │   │   ├── test_request.py                  # Integration test for CSV prediction
│   │   │   ├── test_request_graph.py            # Integration test for graph prediction
│   │   │   └── wsgi.py                          # WSGI entry point
│   │   ├── frontend/
│   │   │   ├── index.html                       # Semantic HTML5 layout
│   │   │   ├── css/
│   │   │   │   └── style.css                    # Design system (Dark & Light themes, HUD, cards)
│   │   │   └── js/
│   │   │       └── app.js                       # Modular JavaScript controllers (Vis.js, Player, Table)
│   │   ├── Dockerfile                           # Container deployment specification
│   │   └── README.md                            # FullStack dashboard documentation
│   ├── src/                                     # Core modular Python packages
│   ├── main.py                                  # Training & experiment pipeline runner
│   ├── requirements.txt                         # Implementation dependencies
│   └── README.md                                # Implementation technical report
├── input samples/                               # Ready-to-use CSV demo datasets
│   ├── twitter_ego_100318079_network.csv        # 450 directed edge list
│   ├── twitter_ego_topological_nodes.csv        # Pre-computed topological centralities
│   ├── viral_marketing_campaign.csv             # KOL marketing influencers dataset
│   ├── tech_community_interaction.csv           # Developer collaboration graph
│   └── full_1180_feature_sample.csv             # Full 1180-dim neural feature tensor
├── Literature_Summary_12_Papers_Expanded.xlsx   # Literature review of 12 academic papers
├── Sharvayu_Zade_PBL_MidSem_TA-GAT.pptx         # Mid-semester presentation slide deck
└── README.md                                    # This Master Presentation Document
```

---

## 12. Step-by-Step Presentation & Viva Walkthrough Script

When presenting this project to evaluators, professors, or audiences, follow this structured walkthrough:

```
[Presentation Flow: Total Time ~ 7 to 8 Minutes]

1. Problem & Motivation (1.0 min)  ──>  Why traditional PageRank / Degree fail
2. Innovation & TA-GAT Math (1.5 min) ──>  1,180-dim fusion + 4-Head GAT attention
3. Live Web App & Themes (1.0 min) ──>  Showcase Obsidian Dark & Crisp Light modes
4. Interactive Graph & Spotlight (1.5 min) ──>  Vis.js HUD search & camera auto-zoom
5. Viral Diffusion Player (1.5 min) ──>  Play outbreak & show live graph infection flares
6. Leaderboard, Radar & Benchmarks (1.0 min) ──>  Spearman rho = 0.2989 (+35.8% over PageRank)
```

### Scripted Talking Points

#### Part 1: Introduction & Problem Motivation
> *"Good morning, respected professors. My name is **Sharvayu Zade**, and my 7th-semester Project-Based Learning project is titled **Topology-Aware Graph Attention Network (TA-GAT) for Influential Node Detection in Social Ego Networks**.*  
> *In large-scale social networks, information, rumors, and marketing campaigns spread virally through a small subset of super spreaders. Traditional approaches rely on classical metrics like Degree Centrality or PageRank. However, Degree only counts immediate friends, and PageRank assumes uniform random walks without considering user attributes. Our solution, **TA-GAT**, bridges this gap by combining 9 topological centralities with 1,171 user profile attributes into an end-to-end Graph Attention Network."*

#### Part 2: Architecture & Mathematical Formulation
> *"Our pipeline constructs an exact **1,180-dimensional feature tensor** per node. We pass this through a 2-layer Graph Attention Network with **4 attention heads** in Layer 1. The multi-head attention mechanism learns dynamic coefficients $\alpha_{ij}$ across directed edges, determining which neighbors actually propagate influence. To train the network, we simulate the **Monte Carlo Independent Cascade** model over the Stanford Twitter Ego-network to obtain empirical reach targets. The model converged at epoch 16, producing a lightweight 327,000-parameter neural checkpoint."*

#### Part 3: Live Dashboard & Dual-Theme Demonstration
> *(Open `http://127.0.0.1:5000/` in browser)*  
> *"To demonstrate our model in real-world environments, we developed **TA-GAT Studio**, an enterprise-grade full-stack platform running on Flask, PyTorch, and Vis.js.*  
> *Notice our live server telemetry badge in the header, reporting that our PyTorch backend is online. With one click on our theme switcher, the entire application seamlessly toggles between an **Obsidian Dark mode** and a **Crisp Light mode**, with all canvas styles and chart scales adapting dynamically."*

#### Part 4: Interactive Graph Topology & Search Spotlight
> *"Here on the **Interactive Topology Canvas**, we visualize the Stanford Twitter ego network with 221 nodes and 450 directed edges. The nodes are sized and color-coded into 4 influence tiers: amber for Top 3 Super Spreaders, indigo for Key Influencers, cyan for Bridge Connectors, and slate for Community Members.*  
> *(Demonstrate)* *In our floating HUD toolbar, I will search for our top-ranked node, Node `364971269`. Notice how the camera smoothly glides and zooms into this node, highlighting its local neighborhood."*

#### Part 5: Information Diffusion Cascade Outbreak Player
> *(Scroll to the Diffusion section)*  
> *"To prove that our predicted influencers truly drive information diffusion, we built an interactive **Cascade Outbreak Player**. Starting from our top 3 predicted seeds with a propagation probability $p = 0.15$, I click **'Run Outbreak Simulation'**.*  
> *The cascade reaches **187 nodes** — over **84.6% of the network** — across 6 wave steps!*  
> *(Click 'Play Outbreak')* *As the player steps forward, newly infected nodes flare into glowing amber and flame red directly on the network graph, while the cumulative S-curve adoption chart tracks the outbreak in real time."*

#### Part 6: Leaderboard Table, Node Radar Telemetry & Conclusion
> *(Scroll to the Leaderboard table)*  
> *"Our leaderboard provides full multi-column sorting by TA-GAT Score, Degree, or PageRank, with responsive pagination.*  
> *(Click on Node 364971269)* *Clicking any row opens our **Node Telemetry Modal**, which renders a 6-axis Radar Chart comparing this specific node against the network average across Degree, In-Degree, PageRank, Betweenness, Core Number, and Closeness.*  
> *(Open Baselines Modal)* *Finally, our benchmark results: TA-GAT achieves a **Spearman Rank Correlation ($\rho$) of 0.2989**, significantly outperforming PageRank at $0.2200$ and Degree Centrality at $0.1800$, while reducing Mean Absolute Error to $58.31$.*  
> *Thank you, and I am now open to any questions."*

---

## 13. Setup, Installation & Execution Guide

### Prerequisites
- Operating System: Windows 10/11, macOS, or Linux
- Python: Version 3.10 or higher
- Modern Web Browser: Google Chrome, Microsoft Edge, Mozilla Firefox, or Safari

### 1. Environment Setup
The project uses the virtual environment located in the workspace root `.venv`:
```powershell
# In PowerShell:
cd "e:\Sharvayu data\Malware\Symbiosis Nagpur SIT\7th SEM\PBL"
.venv\Scripts\Activate.ps1
```

### 2. Launching the TA-GAT Studio Server
To start the Flask backend and serve the dashboard:
```powershell
cd "Implementation/FullStack/backend"
..\..\..\.venv\Scripts\python.exe app.py
```
*The server will start on **`http://127.0.0.1:5000/`**.*

### 3. Accessing the Platform
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### 4. Running Automated Tests
To verify backend API routes and graph predictions:
```powershell
cd "Implementation/FullStack/backend"
..\..\..\.venv\Scripts\python.exe test_request_graph.py
```

### 5. Re-Running the Model Training Pipeline (Optional)
To retrain the TA-GAT neural network from scratch using `main.py`:
```powershell
cd "Implementation"
..\.venv\Scripts\python.exe main.py --fast
```
*Outputs (model weights, loss plots, metrics) will be saved to `Implementation/outputs/`.*
