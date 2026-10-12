from pathlib import Path
import json
import random
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import pandas as pd

# Paths
implementation_dir = Path(r"e:/Sharvayu data/Malware/Symbiosis Nagpur SIT/7th SEM/PBL/Implementation")
outputs_dir = implementation_dir / "outputs"
models_dir = outputs_dir / "models"

# Load first ego graph
def load_first_ego_graph(dataset_subdir: Path):
    data_dir = Path(dataset_subdir)
    edge_files = sorted(data_dir.glob("*.edges"))
    if not edge_files:
        raise FileNotFoundError(f"No .edges files in {data_dir}")
    edge_file = edge_files[0]
    node_id = int(edge_file.stem)
    G = nx.DiGraph()
    G.add_node(node_id)
    with open(edge_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            line=line.strip()
            if not line: continue
            a,b = map(int, line.split())
            G.add_edge(a,b)
    G.add_node(node_id)
    feat_file = edge_file.with_suffix('.feat')
    featnames_file = edge_file.with_suffix('.featnames')
    if feat_file.exists() and featnames_file.exists():
        rows=[]
        for line in open(feat_file,'r',encoding='utf-8'):
            rows.append([int(v) for v in line.strip().split()])
        features = np.array(rows, dtype=np.float32)
        graph_node_ids = sorted(G.nodes())
        if features.shape[0] != len(graph_node_ids):
            if features.shape[0] < len(graph_node_ids):
                pad = np.zeros((len(graph_node_ids)-features.shape[0], features.shape[1]), dtype=np.float32)
                features = np.vstack([features, pad])
            else:
                features = features[:len(graph_node_ids)]
        node_ids = graph_node_ids
    else:
        features = np.eye(len(G.nodes()), dtype=np.float32)
        node_ids = list(G.nodes())
    return G, node_ids, features

# Topological features
from sklearn.preprocessing import StandardScaler

def compute_topological_features(graph: nx.DiGraph):
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    pagerank = nx.pagerank(graph, alpha=0.85)
    try:
        betweenness = nx.betweenness_centrality(graph, k=min(200, graph.number_of_nodes()), normalized=True)
    except Exception:
        betweenness = {n:0.0 for n in graph.nodes()}
    closeness = nx.closeness_centrality(graph)
    clustering = nx.clustering(graph)
    core = nx.core_number(graph)
    try:
        eigen = nx.eigenvector_centrality_numpy(graph, max_iter=1000)
    except Exception:
        eigen = {n:0.0 for n in graph.nodes()}
    rows=[]
    for node in sorted(graph.nodes()):
        rows.append([float(graph.degree(node)), float(in_degree[node]), float(out_degree[node]), float(pagerank[node]), float(betweenness[node]), float(closeness[node]), float(clustering[node]), float(core[node]), float(eigen[node])])
    arr = np.array(rows, dtype=np.float32)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(arr)
    names=['degree','in_degree','out_degree','pagerank','betweenness','closeness','clustering','core_number','eigenvector']
    return scaled, names

# Model
class TAGAT(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, heads: int = 4, dropout: float = 0.3, output_dim: int = 1) -> None:
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)
        self.fc1 = torch.nn.Linear(hidden_dim, 32)
        self.fc2 = torch.nn.Linear(32, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).squeeze(-1)
        return x


def main():
    # dataset is stored at the workspace root `Dataset/` directory
    dataset_subdir = implementation_dir.parent / 'Dataset' / 'Ego_Twitter Stanford University' / 'twitter' / 'twitter'
    G, node_ids, raw_features = load_first_ego_graph(dataset_subdir)
    topo, topo_names = compute_topological_features(G)
    feature_matrix = np.hstack([topo, raw_features[:topo.shape[0]]])
    node_list = list(G.nodes())
    node_to_idx = {n:i for i,n in enumerate(node_list)}
    edge_pairs = [(node_to_idx[u], node_to_idx[v]) for u,v in G.edges()]
    edge_index = np.array(edge_pairs, dtype=np.int64).T if G.number_of_edges()>0 else np.zeros((2,0), dtype=np.int64)
    data = Data(x=torch.tensor(feature_matrix, dtype=torch.float32), edge_index=torch.tensor(edge_index, dtype=torch.long), y=torch.tensor(np.zeros(len(node_list), dtype=np.float32)))

    model_files = sorted(models_dir.glob('*.pt'))
    if not model_files:
        print('No model files found in', models_dir)
        return
    latest = model_files[-1]
    print('Loading model from', latest)
    model = TAGAT(input_dim=data.x.shape[1])
    state = torch.load(latest, map_location='cpu')
    try:
        model.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        preds = model(data).cpu().numpy()
    df = pd.DataFrame({'node_id': node_list, 'pred': preds})
    df.sort_values('pred', ascending=False, inplace=True)
    out_path = outputs_dir / 'rankings' / 'interface_rankings.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print('Saved predictions to', out_path)

if __name__ == '__main__':
    main()
