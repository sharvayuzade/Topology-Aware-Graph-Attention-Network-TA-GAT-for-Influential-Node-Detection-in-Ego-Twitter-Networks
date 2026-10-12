from pathlib import Path

import numpy as np

from src.data_loader import EgoTwitterDataset
from src.diffusion import independent_cascade
from src.graph_analysis import compute_topological_features
from src.models import TAGAT
from src.train import build_graph_data


def test_dataset_loading_and_graph_summary():
    root = Path(__file__).resolve().parents[1].parent
    dataset = EgoTwitterDataset(dataset_dir=root / "Dataset" / "Ego_Twitter Stanford University", subdir=root / "Dataset" / "Ego_Twitter Stanford University" / "twitter" / "twitter")
    graph, node_ids, features = dataset.load_graph()
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert len(node_ids) > 0
    assert features.shape[0] == graph.number_of_nodes()


def test_topological_features_shape():
    root = Path(__file__).resolve().parents[1].parent
    dataset = EgoTwitterDataset(dataset_dir=root / "Dataset" / "Ego_Twitter Stanford University", subdir=root / "Dataset" / "Ego_Twitter Stanford University" / "twitter" / "twitter")
    graph, _, _ = dataset.load_graph()
    features, info = compute_topological_features(graph)
    assert features.shape[0] == graph.number_of_nodes()
    assert features.shape[1] == len(info["feature_names"])


def test_diffusion_smoke():
    root = Path(__file__).resolve().parents[1].parent
    dataset = EgoTwitterDataset(dataset_dir=root / "Dataset" / "Ego_Twitter Stanford University", subdir=root / "Dataset" / "Ego_Twitter Stanford University" / "twitter" / "twitter")
    graph, _, _ = dataset.load_graph()
    spread = independent_cascade(graph, list(graph.nodes())[0], p=0.1, max_steps=5, seed_value=42)
    assert spread >= 1


def test_model_forward_pass():
    data = build_graph_data(np.ones((4, 3), dtype=np.float32), np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64), np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    model = TAGAT(input_dim=3, hidden_dim=8, heads=2, dropout=0.0)
    pred = model(data)
    assert pred.shape == (4,)
