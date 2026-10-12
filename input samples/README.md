# Input Samples Directory: Network Influence Datasets

This directory contains diverse, authentic sample datasets designed for demonstration, testing, and evaluation with the **TA-GAT (Topology-Aware Graph Attention Network)** Influential Node Detection platform.

---

## Dataset Overview & Catalog

| File Name | Format | Rows / Size | Type | Primary Description |
| :--- | :--- | :--- | :--- | :--- |
| **`twitter_ego_100318079_network.csv`** | CSV | 450 Edges | Directed Edge List | Authentic follower/following directed edges from Stanford Ego-Twitter user `100318079`. Ideal for graph topology reconstruction & interactive network visualization. |
| **`twitter_ego_topological_nodes.csv`** | CSV | 220 Nodes | Node Features | Complete set of 9 graph structural metrics (Degree, In/Out-Degree, PageRank, Betweenness, Closeness, Clustering, Core Number, Eigenvector Centrality). |
| **`viral_marketing_campaign.csv`** | CSV | 50 Nodes | Tabular + Metadata | Key Opinion Leader (KOL) and brand ambassador dataset with user handles, influencer tiers (Macro, Bridge, Micro), and reach metrics. |
| **`tech_community_interaction.csv`** | CSV | 125 Edges | Multi-relational Edges | Open-source developer interaction network (code reviews, mentorship, mentions) across active engineering hubs. |
| **`full_1180_feature_sample.csv`** | CSV | 40 Nodes | 1180 Neural Features | Exact feature schema matching the trained PyTorch Geometric TA-GAT model (`best_model.pt`), incorporating raw user attributes and topological vectors. |

---

## Schema Documentation

### 1. `twitter_ego_100318079_network.csv`
- `source_node_id`: Unique numeric Twitter ID of the source user.
- `target_node_id`: Unique numeric Twitter ID of the followed target user.
- `relationship`: Type of directed tie (`follows`).

### 2. `twitter_ego_topological_nodes.csv`
- `node_id`: Unique numeric Twitter ID.
- `degree`: Total connected incident edges ($k_{in} + k_{out}$).
- `in_degree`: Total incoming follower edges (direct authority measure).
- `out_degree`: Total outgoing following edges (activity indicator).
- `pagerank`: Stationary random walk probability ($\alpha = 0.85$).
- `betweenness`: Fraction of shortest paths traversing through this node.
- `closeness`: Reciprocal of the average shortest path distance to all other nodes.
- `clustering`: Local clustering coefficient (triadic closure measure).
- `core_number`: Highest $k$-core subgraph membership.
- `eigenvector`: Dominant eigenvector centrality in the adjacency matrix.

### 3. `viral_marketing_campaign.csv`
- `node_id`: Unique identifier for the social account.
- `user_handle`: Social username alias (e.g., `user_1042`).
- `community`: Topical cluster (`Tech Influencers`, `AI Researchers`, `Growth Hackers`, etc.).
- `tier`: Strategic classification (`Tier-1 Key Opinion Leader`, `Community Bridge`, `Micro Influencer`).
- `role`: Social functional role (`Macro Influencer`, `Community Moderator`, `Active Contributor`).
- `degree`, `in_degree`, `out_degree`, `pagerank`, `betweenness`, `closeness`, `clustering`, `core_number`, `eigenvector`: Topological connectivity metrics.

### 4. `tech_community_interaction.csv`
- `source_dev`: Identifier of the initiating developer.
- `target_dev`: Identifier of the receiving developer.
- `interaction_type`: Interaction classification (`code_review`, `mentorship`, `issue_discussion`, `mention`, `retweet`).
- `weight`: Interaction intensity weight.

### 5. `full_1180_feature_sample.csv`
- `node_id`: Unique node ID.
- `degree`, `in_degree`, `out_degree`, `pagerank`, `betweenness`, `closeness`, `clustering`, `core_number`, `eigenvector`: Standard 9 topological features.
- `node_feat_0` ... `node_feat_1170`: 1171 high-dimensional semantic and profile attributes extracted from the Stanford Twitter ego files.

---

## How to Test in the Dashboard

1. Launch the FullStack server:
   ```bash
   cd Implementation/FullStack/backend
   python app.py
   ```
2. Open `http://127.0.0.1:5000/` in your browser.
3. In the **"Sample Presets Gallery"**, click any of the dataset cards:
   - Click **"Twitter Ego Network"** to load the directed graph, visualize the interactive topology, and run the GAT model.
   - Click **"Viral Marketing KOLs"** to view influence rankings grouped by community and tier.
   - Click **"Simulate Cascade"** to launch an animated Independent Cascade diffusion propagation from the highest-ranked seed nodes.
