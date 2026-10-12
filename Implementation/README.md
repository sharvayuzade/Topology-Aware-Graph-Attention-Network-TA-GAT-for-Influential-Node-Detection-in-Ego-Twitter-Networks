# TA-GAT for Influential Node Detection in Ego Twitter Networks

## 1. Project Summary
This project implements a graph-based machine learning pipeline for identifying influential nodes in a social network, specifically using the Stanford Ego Twitter dataset. The goal is to estimate which users are most likely to influence others in the network based on both structural topology and learned graph representations.

The core model is a Topology-Aware Graph Attention Network (TA-GAT), which combines graph topology features (degree, PageRank, betweenness, closeness, clustering, etc.) with a graph neural network architecture. The project also compares the learned model against traditional network baselines such as degree centrality, betweenness centrality, closeness centrality, and PageRank.

In practical terms, the system answers the question:

"Given a social graph, which users are likely to be the most influential spreaders or opinion leaders?"

This is relevant for understanding online influence, information diffusion, viral campaigns, social recommendation, and network analysis.

---

## 2. Problem Statement
Social networks are not just collections of nodes; they are dynamic influence systems. Some users shape conversations, trigger cascades, and connect communities. Traditional graph metrics can identify topologically important nodes, but they may fail to capture complex relational patterns, hidden interactions, and non-linear influence propagation.

The project addresses this by:

- reading the directed Twitter ego network,
- extracting structural and semantic node features,
- computing influence targets through diffusion simulation,
- training a graph attention model to learn influence patterns,
- ranking nodes by predicted influence,
- comparing with conventional centrality methods.

This makes the project a strong academic and applied demonstration of graph neural networks in network influence analysis.

---

## 3. Project Objectives
The main objectives of the implementation are:

1. Load and preprocess ego-network social data from Stanford's Twitter public dataset.
2. Construct a directed graph using follower/following relationships.
3. Extract topology-aware node features from graph structure.
4. Generate influence labels using an Independent Cascade simulation.
5. Train a graph attention model to predict node influence.
6. Evaluate performance using regression and ranking metrics.
7. Compare model predictions against classical baseline ranking methods.
8. Produce reusable scientific outputs such as rankings, metrics, and visual plots.

---

## 4. Why This Project Matters
Influential node detection has numerous practical applications:

- marketing and campaign targeting,
- misinformation and rumor control,
- recommendation systems,
- community detection and epidemic spread analysis,
- social influence measurement,
- organizational communication strategy,
- network resilience and vulnerability analysis.

Classic centrality metrics are useful but often insufficient because they do not learn from rich feature interactions. A GNN-based model can capture local neighborhood context and attention-based weighting, which makes it more expressive for influence estimation.

---

## 5. Dataset Overview
The project uses the Stanford Ego Twitter dataset provided in the workspace under:

- Dataset/Ego_Twitter Stanford University/
- Dataset/Ego_Twitter Stanford University/twitter/twitter/

### Dataset characteristics
Each ego network folder typically contains files such as:

- .edges: directed edges between nodes in the ego network
- .feat: node feature matrix
- .featnames: names of feature attributes
- .egofeat: ego-user feature vector
- .circles: community or group membership labels

### Why this dataset is suitable
- It is a real-world social network dataset.
- It is a standard benchmark in network analysis research.
- It contains rich local structure and node attributes.
- It is small enough for experiments in a student project and research prototype.
- It supports both topology-based analysis and graph learning.

### Important modeling decision
The project treats the graph as directed because Twitter edges represent a following relationship, which is inherently directional. This is important because influence flow is not symmetric in social media.

---

## 6. Project Workflow
The pipeline follows a clear sequence:

1. Dataset discovery and graph loading
2. Topological feature extraction
3. Optional feature normalization
4. Influence label generation
5. Graph conversion to PyTorch Geometric format
6. Model training and validation
7. Prediction and ranking generation
8. Baseline comparison and metric saving
9. Output generation (plots, CSVs, model checkpoints)

### Core mathematical idea
The goal is to learn a function:

f(G, X) -> influence_score

where:

- G is the directed graph,
- X is the node feature matrix,
- influence_score estimates a node's expected spread or impact.

The model is trained to approximate a target signal generated by Independent Cascade simulation, which acts as a proxy for influence propagation.

---

## 7. Methodology
### 7.1 Topological feature extraction
The implementation computes several graph features for each node, including:

- degree
- in-degree
- out-degree
- PageRank
- betweenness centrality
- closeness centrality
- clustering coefficient
- core number
- eigenvector centrality

These are useful because they describe different aspects of influence:

- degree reveals connectivity,
- PageRank measures global importance,
- betweenness captures bridging power,
- closeness reflects reachability,
- clustering captures local redundancy,
- core number indicates structural centrality.

### 7.2 Influence target generation
The project uses Independent Cascade (IC) simulation to generate influence labels. In simple terms, the model simulates how information may spread through the network with a propagation probability.

This is valuable because real-world influence is not just based on static connectivity; it depends on diffusion dynamics. The IC model approximates influence spread under probabilistic activation rules.

The configuration uses:

- propagation probability: 0.1
- Monte Carlo simulations: 5
- max steps: 50
- sample size: configurable for fast mode

### 7.3 Graph Attention model
The model is built on GATConv layers from PyTorch Geometric.

Graph Attention Networks are designed to learn which neighboring nodes are more important for each node's representation. This is significantly better than simple average or sum pooling because each neighbor gets an attention weight.

The model architecture in the project follows a pattern like:

- input layer -> GATConv
- activation -> dropout
- second GATConv
- activation -> dropout
- dense layers -> final scalar prediction

This allows the model to focus on important neighbors while still preserving network structure.

---

## 8. Tech Stack and Why It Was Used

### Python
Python is the main language because it has the strongest ecosystem for data science, graph analysis, and deep learning.

Why used:

- easy integration with scientific libraries,
- excellent graph and ML tooling,
- fast prototyping for research workflows,
- compatibility with PyTorch and PyTorch Geometric.

Alternatives:

- R: good for statistics, but weaker for deep graph learning workflows.
- Java: powerful but less convenient for experimentation and scientific prototyping.
- MATLAB: good for research but expensive and less open-source friendly.

Why Python is better here:

- the project combines graph theory, deep learning, and scientific visualization in one ecosystem,
- it is widely supported in research and academia,
- it integrates naturally with Jupyter, PyTorch, scikit-learn, and NetworkX.

### NetworkX
NetworkX is used for graph creation, traversal, and centrality computations.

Why used:

- simple API for network algorithms,
- ideal for fast research prototypes,
- supports directed graphs and many centrality metrics.

Alternatives:

- igraph: more efficient for very large graphs but heavier and less beginner-friendly,
- SNAP: powerful but more specialized and less Pythonic for a quick project.

Why NetworkX is better here:

- it is simple and fast enough for medium-sized social graphs,
- it supports the exact algorithms needed in this project,
- it is highly readable for academic and educational use.

### NumPy and Pandas
These are used for numerical matrix operations and tabular data management.

Why used:

- efficient vectorized computations,
- easy conversion between arrays and DataFrames,
- helps with feature engineering and output generation.

Alternatives:

- Polars: faster for large DataFrames but not as widely used in academic ML workflows.
- MATLAB arrays: strong in numerical work but not as flexible in this Python stack.

Why better here:

- NumPy and Pandas are the default standard,
- they integrate well with PyTorch and scikit-learn,
- they are stable and well documented.

### PyTorch
PyTorch is the deep learning framework used for implementing the model and training loop.

Why used:

- dynamic computation graph,
- strong research adoption,
- straightforward model debugging,
- works smoothly with graph neural networks.

Alternatives:

- TensorFlow: strong but often more verbose and less research-friendly for experimentation,
- JAX: powerful but less familiar and more specialized.

Why PyTorch is better here:

- easy to prototype quickly,
- natural for custom graph learning logic,
- strong community support and integration with PyG.

### PyTorch Geometric (PyG)
PyTorch Geometric is used for graph neural network processing and message passing.

Why used:

- built for graph data structures,
- implements GATConv and other graph operators,
- integrates naturally with PyTorch.

Alternatives:

- DGL: also strong, but PyG is commonly preferred for GNN research and graph attention models,
- custom manual graph training loops: possible but much slower and error-prone.

Why PyG is better here:

- simpler graph representation for node classification and regression tasks,
- direct access to graph attention layers,
- better fit for this project's architecture and learning objective.

### scikit-learn
Used for preprocessing, metrics, and baseline comparison.

Why used:

- standardized metrics,
- simple train/validation split utilities,
- easy integration with data pipelines.

Alternatives:

- XGBoost or LightGBM: powerful for tabular data, but not directly graph-native,
- custom metrics implementations: possible but unnecessary.

Why better here:

- standard, proven, and lightweight,
- ideal for benchmarking and evaluation in a graph ML project.

### Matplotlib
Used to generate plots such as loss curves and prediction-vs-actual scatter plots.

Why used:

- simple and widely used,
- sufficient for publication-quality research plots,
- easy to integrate with standard Python workflows.

Alternatives:

- Seaborn: better aesthetics but less direct control,
- Plotly: interactive, but heavier for a simple local research project.

Why better here:

- minimal dependencies,
- enough for analysis and reporting,
- good compatibility with static outputs.

### YAML Configuration
The project uses YAML files for configuration management.

Why used:

- readable and human-friendly,
- easy to adjust hyperparameters without editing code,
- suitable for academic prototypes.

Alternatives:

- JSON: fine but less readable,
- command-line args only: less manageable for many settings.

Why better here:

- easy experimentation,
- centralizes project settings,
- keeps the code cleaner and more reusable.

### TA-GAT Studio: Production FullStack Intelligence Platform
The `FullStack` folder contains **TA-GAT Studio**, an industry-grade web application with a Flask REST API backend, PyTorch Geometric inference engine, and an interactive frontend dashboard.

Key Capabilities:
- **Dual-Theme Engine:** Obsidian Dark Mode (default) & Crisp Light Mode with instant toggle and state persistence.
- **Interactive Vis.js Graph Canvas:** Real-time force-directed topology, search & focus spotlight camera zoom, tier-based color hierarchy (Super Spreaders, Key Influencers, Bridge Connectors), and physics freeze.
- **Animated Information Diffusion Cascade Player:** Independent Cascade (IC) epidemic simulation with step-by-step playback controls (Play, Pause, Step, Reset) and real-time visual infection flare on the network graph.
- **Sortable & Paginated Leaderboard:** Multi-column sorting, pagination, and row-click inspection with a 6-axis Radar comparison chart.
- **System Telemetry:** Real-time `/api/health` monitoring PyTorch version, active model, and device (`cpu` / `cuda`).

Refer to the dedicated presentation guide: [`FullStack/README.md`](FullStack/README.md).

---

## 9. Why This Stack Is Better Than Other Approaches
This project is intentionally designed as a graph learning system rather than a pure classic network analysis pipeline. The chosen stack is better than alternatives because:

### Compared with pure centrality-based methods
Traditional measures such as PageRank and betweenness are fast and interpretable, but they fail to learn from node features and local neighborhood context. TA-GAT can capture patterns beyond simple structural rules.

### Compared with classical machine learning only
A logistic regression or Random Forest model might perform reasonably on engineered features, but it does not naturally operate on graph structure or neighbor interactions.

### Compared with fully custom GNN frameworks
Using PyG reduces implementation risk and improves correctness. A custom implementation would be slower, harder to debug, and more error-prone.

### Compared with heavy software stacks
A full enterprise stack (e.g., Node.js + React + database + API) would be unnecessary overhead for a research dataset and one-model pipeline. The chosen stack keeps the project focused and easier to run.

---

## 10. File and Folder Structure
The project is organized as follows:

- Implementation/
  - main.py: main training and evaluation pipeline
  - README.md: project documentation
  - requirements.txt: package dependencies
  - configs/config.yaml: experiment parameters
  - outputs/: generated model artifacts, rankings, logs, and figures
  - FullStack/: Flask + frontend demo
  - Notebooks/: exploratory Jupyter workflow
  - tests/: validation smoke tests

### Important execution flow
The training entry point runs from:

- Implementation/main.py

It loads configuration from:

- Implementation/configs/config.yaml

It uses the dataset under:

- Dataset/Ego_Twitter Stanford University/

It writes results under:

- Implementation/outputs/

---

## 11. Configuration Details
The configuration file contains important project parameters:

- seed: 42
- device: auto
- fast_mode: true
- full_experiment: false
- regenerate_targets: false

Model hyperparameters include:

- hidden_dim: 64
- heads: 4
- dropout: 0.3
- learning rate: 0.005
- weight decay: 5e-4
- epochs: 20
- patience: 8

Feature settings:

- use_topological_features: true
- use_node_features: true
- normalize: true

Influence simulation settings:

- propagation_probability: 0.1
- monte_carlo_sims: 5
- max_steps: 50
- sample_nodes: 200

These values are tuned to balance computational load, model quality, and reproducibility.

---

## 12. Training and Inference Pipeline
### Training
The main training process does the following:

1. Load ego network and features.
2. Build topological feature matrix.
3. Combine topology features with raw node features when enabled.
4. Normalize features.
5. Compute influence labels via Independent Cascade.
6. Convert node graph to PyG Data object.
7. Train the model using a prediction objective relevant to influence score estimation.
8. Save checkpoint, metrics, and rankings.

### Inference
Once training is complete, the model predicts influence scores for every node. These scores are then:

- sorted descending,
- saved in CSV rankings,
- compared with classical baseline rankings,
- stored in model metrics files.

---

## 13. Evaluation Strategy
The project evaluates both regression and ranking quality.

### Regression metrics
These may include metrics such as:

- MSE
- MAE
- RMSE
- correlation-based assessment

### Ranking metrics
It also compares node ordering using ranking metrics, which is highly relevant for identifying influential nodes.

### Baseline comparison
The system compares TA-GAT predictions to methods such as:

- degree centrality,
- in-degree,
- out-degree,
- betweenness,
- PageRank,
- closeness,
- clustering,
- core number.

This comparison is essential because a graph model should outperform naive graph statistics in a meaningful way.

---

## 14. Outputs Produced by the Project
The outputs directory contains all key artifacts:

- outputs/models/: trained model checkpoints
- outputs/metrics/: influence labels and model metrics
- outputs/rankings/: baseline and TA-GAT rankings in CSV format
- outputs/figures/: training loss and prediction plots
- outputs/logs/: execution logs

These outputs make the project useful not only for training but also for reporting and analysis.

---

## 15. How to Run the Project
### Install dependencies
From the Implementation folder:

```bash
pip install -r requirements.txt
```

### Run the main pipeline
```bash
python main.py --fast
```

### Full experiment mode
```bash
python main.py --full
```

### Regenerate influence targets
```bash
python main.py --regenerate-targets
```

---

## 16. Recommended Environment
The project works well in the following environment:

- Windows or Linux
- Python 3.10+
- CUDA-enabled GPU recommended for faster training, though CPU is possible
- sufficient RAM for medium-size graph datasets

---

## 17. Practical Notes and Project Status
The project is structured as a research-style prototype with reproducible outputs and modular organization. It is especially suitable for:

- academic demonstration,
- graph learning coursework,
- network analysis studies,
- prototype deployment in research labs,
- further extension into larger social network analysis tasks.

The implementation already includes a lightweight interface and dashboard prototypes in the FullStack folder, which makes it easier to demonstrate model predictions in a browser-based format.

---

## 18. Strengths of the Project
- uses real social graph data,
- combines topology and graph learning,
- includes influence diffusion simulation,
- compares against centrality baselines,
- modular and understandable structure,
- exports evaluation artifacts,
- supports both CPU and GPU-style experimentation,
- has a demo interface for usability testing.

---

## 19. Limitations and Future Improvements
Although the current implementation is strong, there are still avenues for improvement:

- larger and more diverse social datasets could be used,
- full hyperparameter tuning could improve model performance,
- more advanced diffusion models could replace basic Independent Cascade approximations,
- better handling of large graph memory and runtime issues,
- more formal comparison with other GNN variants such as GCN, GraphSAGE, and Graph Transformer,
- addition of explainability to show why a node is influential,
- deployment as a web application with richer dashboards.

---

## 20. Alternative Models Considered
Other graph learning approaches were possible, including:

- GCN: simpler and efficient, but less expressive in neighbor attention weighting,
- GraphSAGE: useful for inductive learning, but less targeted for attention-based influence modeling,
- PageRank only: fast and interpretable, but not learned and not feature-aware,
- Logistic regression on engineered features: easier to train but graph-agnostic,
- Heterogeneous graph models: unnecessarily complex for this ego-network use case.

The reason TA-GAT was selected is that it balances structure learning, attention, and interpretability while still being practical within a student research project.

---

## 21. Final Conclusion
This project is a complete graph intelligence pipeline for influential node detection in social networks. It combines network science, diffusion simulation, and graph neural networks into a single practical workflow. The implementation is robust enough to serve as an academic project, research prototype, or demonstration application.

The chosen technology stack is appropriate because it balances performance, readability, scientific trust, and ease of experimentation. By combining classical centrality methods with graph attention learning, the project succeeds in representing both traditional network theory and modern deep learning approaches.

---

## 22. Quick Start Summary
If you want the shortest path to running the project:

1. Open the Implementation folder.
2. Install packages from requirements.txt.
3. Run Python main.py --fast.
4. Inspect the generated outputs in outputs/.
5. Use the FullStack demo if you want a browser-based demonstration.

---

## 23. Credits and Context
This work is best understood as a social network intelligence project that blends:

- graph theory,
- influence propagation simulation,
- deep graph learning,
- model evaluation,
- visualization,
- applied AI for network analysis.

It is a strong example of how machine learning can support real-world decision making in social and information networks.
