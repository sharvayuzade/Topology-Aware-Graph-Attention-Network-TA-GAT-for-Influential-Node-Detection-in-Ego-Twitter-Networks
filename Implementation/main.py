from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from src.baselines import compute_baseline_scores, make_baseline_rankings
from src.data_loader import EgoTwitterDataset
from src.diffusion import compute_influence_targets
from src.evaluate import compute_ranking_metrics, compute_regression_metrics, generate_rankings, save_metrics
from src.graph_analysis import compute_topological_features, summarize_features
from src.models import TAGAT
from src.train import build_graph_data, train_model
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger
from src.visualization import save_feature_heatmap, save_loss_plot, save_scatter_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="TA-GAT influential node detection")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--regenerate-targets", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.fast:
        config["project"]["fast_mode"] = True
    if args.full:
        config["project"]["full_experiment"] = True
        config["project"]["fast_mode"] = False

    implementation_dir = Path(__file__).resolve().parent
    project_root = implementation_dir.parent
    outputs_dir = implementation_dir / config["paths"]["outputs_dir"]
    ensure_dir(outputs_dir)
    logger = setup_logger(outputs_dir / "logs" / "run.log")
    set_seed(config["project"]["seed"])

    dataset_dir = project_root / config["paths"]["dataset_dir"]
    subdir = project_root / config["paths"]["dataset_subdir"]
    dataset = EgoTwitterDataset(dataset_dir=dataset_dir, subdir=subdir)
    graph, node_ids, raw_features = dataset.load_graph()
    logger.info("Loaded graph with %s nodes and %s edges", graph.number_of_nodes(), graph.number_of_edges())

    topo_features, feature_info = compute_topological_features(graph)
    assert topo_features.shape[0] == len(node_ids)

    if config["features"]["use_node_features"]:
        feature_matrix = np.hstack([topo_features, raw_features[: topo_features.shape[0]]])
    else:
        feature_matrix = topo_features

    feature_names = feature_info["feature_names"] + [f"node_feat_{i}" for i in range(raw_features.shape[1])]
    if config["features"]["normalize"]:
        std = feature_matrix.std(axis=0)
        std[std == 0.0] = 1.0
        feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / std

    if config["influence"]["cache_targets"] and not args.regenerate_targets:
        targets_path = outputs_dir / "metrics" / "influence_targets.json"
        if targets_path.exists():
            with open(targets_path, "r", encoding="utf-8") as handle:
                target_values = json.load(handle)
                influence_targets = {int(k): float(v) for k, v in target_values.items()}
            logger.info("Loaded cached influence targets from %s", targets_path)
        else:
            influence_targets = compute_influence_targets(graph, p=config["influence"]["propagation_probability"], sims=config["influence"]["monte_carlo_sims"], max_steps=config["influence"]["max_steps"], sample_size=config["influence"]["sample_nodes"] if config["project"]["fast_mode"] else None, seed=config["project"]["seed"])
            with open(targets_path, "w", encoding="utf-8") as handle:
                json.dump(influence_targets, handle, indent=2)
    else:
        influence_targets = compute_influence_targets(graph, p=config["influence"]["propagation_probability"], sims=config["influence"]["monte_carlo_sims"], max_steps=config["influence"]["max_steps"], sample_size=config["influence"]["sample_nodes"] if config["project"]["fast_mode"] else None, seed=config["project"]["seed"])
        targets_path = outputs_dir / "metrics" / "influence_targets.json"
        with open(targets_path, "w", encoding="utf-8") as handle:
            json.dump(influence_targets, handle, indent=2)

    # Align targets with node order and remap graph node IDs to contiguous indices.
    node_list = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    y = np.array([influence_targets.get(node, 0.0) for node in node_list], dtype=np.float32)
    if graph.number_of_edges() > 0:
        edge_pairs = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
        edge_index = np.array(edge_pairs, dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    data = build_graph_data(feature_matrix, edge_index, y)
    model, training_info = train_model(data, config, outputs_dir / "models")
    model.eval()

    with torch.no_grad():
        preds = model(data.to(get_device())).cpu().numpy()

    regression_metrics = compute_regression_metrics(y, preds)
    ranking_metrics = compute_ranking_metrics(y, preds)
    save_metrics({**regression_metrics, **ranking_metrics}, outputs_dir / "metrics" / "model_metrics.json")

    rankings = generate_rankings(node_ids, preds, y, feature_matrix, feature_names)
    rankings.to_csv(outputs_dir / "rankings" / "ta_gat_rankings.csv", index=False)

    baseline_scores = compute_baseline_scores(graph)
    baseline_rankings = make_baseline_rankings(node_ids, baseline_scores)
    for name, df in baseline_rankings.items():
        df.to_csv(outputs_dir / "rankings" / f"{name}_baseline.csv", index=False)

    save_loss_plot(training_info["history"], outputs_dir / "figures" / "training_loss.png")
    save_scatter_plot(y, preds, outputs_dir / "figures" / "predicted_vs_actual.png")
    save_feature_heatmap(feature_matrix, feature_names, outputs_dir / "figures" / "feature_heatmap.png")

    logger.info("Completed TA-GAT pipeline")


if __name__ == "__main__":
    main()
