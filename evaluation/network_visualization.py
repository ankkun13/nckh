"""
network_visualization.py — Trực quan hóa mạng Hub Genes

Vẽ đồ thị mạng lưới làm nổi bật:
    - Các "Hub genes" (gen có độ kết nối cao nhất)
    - Driver genes được xác nhận (màu đỏ)
    - Mạng lưới cục bộ của top-K hub genes

Sử dụng:
    python evaluation/network_visualization.py --config configs/config.yaml \
        --cancer LUAD --subtype 0
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml


def build_networkx_graph(
    adj: sp.csr_matrix,
    gene_list: list[str],
    logger: logging.Logger,
) -> nx.Graph:
    """Chuyển sparse adjacency matrix → NetworkX graph."""
    cx = adj.tocoo()
    G = nx.Graph()
    G.add_nodes_from(gene_list)
    for i, j, w in zip(cx.row, cx.col, cx.data):
        if i < j:
            G.add_edge(gene_list[i], gene_list[j], weight=float(w))
    logger.info(f"[NETX] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def get_hub_genes(G: nx.Graph, top_k: int) -> list[str]:
    """Trả về Top-K gen có degree cao nhất."""
    degrees = dict(G.degree(weight="weight"))
    return sorted(degrees, key=degrees.get, reverse=True)[:top_k]


def plot_hub_gene_network(
    G: nx.Graph,
    hub_genes: list[str],
    driver_genes: set[str],
    predicted_drivers: set[str],
    out_path: Path,
    cancer: str,
    subtype: int,
    logger: logging.Logger,
    layout: str = "spring",
):
    """
    Vẽ subgraph của hub genes với màu sắc:
        - Đỏ đậm: Known driver genes
        - Cam: Predicted new driver genes (novel predictions)
        - Xanh lam: Hub genes non-driver
        - Kích thước node: Tỉ lệ với degree
    """
    # Lấy subgraph: hub_genes + tất cả neighbors trong 1-hop
    nodes_to_include = set(hub_genes)
    for gene in hub_genes:
        if gene in G:
            nodes_to_include.update(list(G.neighbors(gene))[:5])  # Max 5 neighbors/hub

    subG = G.subgraph(list(nodes_to_include)).copy()

    # Loại node cô lập
    subG.remove_nodes_from(list(nx.isolates(subG)))

    if len(subG.nodes) == 0:
        logger.warning("[PLOT] Subgraph rỗng, bỏ qua visualization")
        return

    logger.info(f"[PLOT] Subgraph: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(subG, seed=42, k=2.5 / np.sqrt(subG.number_of_nodes() + 1))
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(subG)
    else:
        pos = nx.circular_layout(subG)

    # Màu sắc nodes
    node_colors = []
    for gene in subG.nodes():
        if gene in driver_genes:
            node_colors.append("#e74c3c")  # Đỏ: known driver
        elif gene in predicted_drivers:
            node_colors.append("#e67e22")  # Cam: predicted driver
        elif gene in hub_genes:
            node_colors.append("#3498db")  # Xanh: hub non-driver
        else:
            node_colors.append("#95a5a6")  # Xám: neighbor

    # Kích thước theo weighted degree
    degrees = dict(subG.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [300 + 1200 * (degrees.get(g, 0) / max_deg) for g in subG.nodes()]

    # Edge weights
    edge_weights = [subG[u][v].get("weight", 0.1) for u, v in subG.edges()]
    edge_alphas = [min(1.0, 0.2 + w * 2) for w in edge_weights]

    # Vẽ
    fig, ax = plt.subplots(figsize=(14, 11))
    nx.draw_networkx_edges(
        subG, pos, ax=ax,
        edge_color=["#bdc3c7"] * len(subG.edges()),
        alpha=0.4, width=0.8,
    )
    nx.draw_networkx_nodes(
        subG, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
    )

    # Labels chỉ cho hub genes và driver genes
    important_genes = set(hub_genes) | (driver_genes & set(subG.nodes()))
    labels = {g: g for g in subG.nodes() if g in important_genes}
    nx.draw_networkx_labels(subG, pos, labels=labels, ax=ax, font_size=7, font_weight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color="#e74c3c", label="Known Driver Gene"),
        mpatches.Patch(color="#e67e22", label="Predicted Novel Driver"),
        mpatches.Patch(color="#3498db", label=f"Hub Gene (Top {len(hub_genes)})"),
        mpatches.Patch(color="#95a5a6", label="Neighbor Gene"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title(
        f"Hub Gene Network — {cancer} Subtype {subtype}\n"
        f"{subG.number_of_nodes()} genes, {subG.number_of_edges()} interactions",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"[SAVE] Network visualization: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="configs/config.yaml")
    parser.add_argument("--cancer",  default="LUAD")
    parser.add_argument("--subtype", default=0, type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cancer        = args.cancer.upper()
    processed_dir = Path(cfg["paths"]["processed"][cancer.lower()])
    labels_dir    = Path(cfg["paths"]["raw"]["labels"])
    fused_dir     = processed_dir / "fused_networks"
    results_dir   = Path(cfg["paths"].get("results", "/workspace/results")) / cancer
    results_dir.mkdir(parents=True, exist_ok=True)
    subtype       = args.subtype
    top_k         = cfg["evaluation"]["network_viz"]["hub_gene_top_k"]
    layout        = cfg["evaluation"]["network_viz"]["layout"]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger = logging.getLogger("netx_viz")

    # Tải fused network
    adj_file  = fused_dir / f"fused_adjacency_subtype{subtype}.npz"
    gene_file = fused_dir / f"fused_genes_subtype{subtype}.txt"
    if not adj_file.exists():
        logger.error(f"Không tìm thấy: {adj_file}")
        return

    adj = sp.load_npz(str(adj_file))
    with open(gene_file) as f:
        gene_list = [line.strip() for line in f if line.strip()]

    G = build_networkx_graph(adj, gene_list, logger)
    hub_genes = get_hub_genes(G, top_k)
    logger.info(f"Top {top_k} hub genes: {hub_genes[:10]}...")

    # Driver gene labels
    driver_file = labels_dir / "driver_gene_set.txt"
    with open(driver_file) as f:
        known_drivers = set(line.strip() for line in f if line.strip())

    # Predictions
    pred_file = results_dir / f"predictions_subtype{subtype}.csv"
    predicted_drivers = set()
    if pred_file.exists():
        pred_df = pd.read_csv(pred_file)
        predicted_new = pred_df[
            (pred_df["driver_probability"] >= 0.5) &
            (~pred_df["gene"].isin(known_drivers))
        ]["gene"].tolist()
        predicted_drivers = set(predicted_new)
        logger.info(f"  -> {len(predicted_drivers)} predicted novel drivers")

    plot_hub_gene_network(
        G, hub_genes,
        driver_genes=known_drivers,
        predicted_drivers=predicted_drivers,
        out_path=results_dir / f"hub_gene_network_subtype{subtype}.png",
        cancer=cancer,
        subtype=subtype,
        logger=logger,
        layout=layout,
    )


if __name__ == "__main__":
    main()
