"""
gnn.py — Kiến trúc Graph Neural Network cho Driver Gene Identification

Kết hợp hai cải tiến từ literature:
    1. Initial Residual Connection (MLGCN-Driver): nối tắt từng GCN layer
       về input ban đầu → chống Over-smoothing
    2. Representation Separation (SGCD): tách biệt omics features và
       graph topology features → chống feature confusion trong Heterophilic graphs

Kiến trúc tổng thể:
    ┌─────────────────────────────────────────────────────┐
    │  Omics Features (RNA, Meth, Mut)                    │
    │        ↓ Linear Encoder                             │
    │  Omics Embedding ──────────────────────┐            │
    │                                        │            │
    │  PPI Topology (node2vec)               │            │
    │        ↓ Linear Encoder                │            │
    │  Topo Embedding  ──────────────────────┤            │
    │                                        ↓            │
    │              ┌── GATConv [Layer 1] ← Initial Resid │
    │              └── GATConv [Layer 2] ← Initial Resid │
    │                      ↓                              │
    │              Final Representation                   │
    │                      ↓                              │
    │              MLP Classifier → sigmoid → P(driver)   │
    └─────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import dropout_adj


class OmicsEncoder(nn.Module):
    """
    Mã hoá omics features (RNA-seq + Methylation + Mutation) thành embedding.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopologyEncoder(nn.Module):
    """
    Mã hoá node2vec topology features thành embedding.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GNNLayer(nn.Module):
    """
    Một lớp GNN (GAT hoặc GCN) với Initial Residual Connection.

    Initial Residual (từ MLGCN-Driver):
        h_l = (1 - alpha) * Conv(h_{l-1}) + alpha * h_0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.5,
        conv_type: str = "gat",
        initial_residual_alpha: float = 0.2,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.alpha = initial_residual_alpha
        self.dropout = dropout
        self.out_channels = out_channels

        if conv_type == "gat":
            # GAT: output dim = out_channels (concat heads rồi reduce về out_channels)
            self.conv = GATConv(
                in_channels=in_channels,
                out_channels=out_channels // heads,
                heads=heads,
                dropout=dropout,
                concat=True,  # concat attention heads → dim = out_channels
            )
            effective_out = out_channels  # = heads * (out_channels // heads)
        elif conv_type == "gcn":
            self.conv = GCNConv(
                in_channels=in_channels,
                out_channels=out_channels,
                improved=True,  # A_hat = A + 2I (tự kết nối mạnh hơn)
            )
            effective_out = out_channels
        else:
            raise ValueError(f"conv_type không hợp lệ: {conv_type}")

        self.norm = nn.LayerNorm(effective_out)

        # Projection cho initial residual (nếu kích thước thay đổi)
        if in_channels != effective_out:
            self.residual_proj = nn.Linear(in_channels, effective_out, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        x0: torch.Tensor,   # Initial input (h_0) cho Initial Residual
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          Node features tại layer hiện tại, shape (N, in_channels)
            edge_index: COO format edges, shape (2, E)
            x0:         Node features ban đầu (input to GNN), shape (N, in_channels_0)
            edge_weight: Edge weights (tuỳ chọn)
        """
        # Dropout trên edges để tăng robustness (tránh overfitting trên graph)
        if self.training and self.dropout > 0:
            edge_index, edge_weight = dropout_adj(
                edge_index, edge_attr=edge_weight,
                p=self.dropout * 0.5,  # Edge dropout nhẹ hơn node dropout
                training=self.training,
            )

        # Graph convolution
        if self.conv_type == "gat":
            h = self.conv(x, edge_index)
        else:
            h = self.conv(x, edge_index, edge_weight=edge_weight)

        # Initial Residual Connection
        h0_proj = self.residual_proj(x0)
        h = (1 - self.alpha) * h + self.alpha * h0_proj

        # Normalize + Activate
        h = self.norm(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class MODCANGNNModel(nn.Module):
    """
    Mô hình GNN chính cho Driver Gene Identification.

    Cải tiến:
        1. Representation Separation (SGCD): Hai nhánh độc lập xử lý
           omics features và topology features.
        2. Initial Residual Connections (MLGCN-Driver): Mỗi GNN layer
           nối tắt về h_0 ban đầu, tránh over-smoothing.
        3. Focal Loss compatible output (logit, không sigmoid).

    Args:
        omics_dim:      Số chiều omics features (RNA + Meth + Mut)
        topo_dim:       Số chiều node2vec embedding
        hidden_channels: Kích thước hidden layer GNN
        num_layers:     Số lớp GNN (tối đa 3, khuyến nghị 2)
        heads:          Số attention heads (chỉ cho GAT)
        dropout:        Dropout rate
        conv_type:      "gat" hoặc "gcn"
        initial_residual: Bật/tắt Initial Residual Connection
        representation_separation: Bật/tắt Representation Separation
    """

    def __init__(
        self,
        omics_dim: int,
        topo_dim: int,
        hidden_channels: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
        conv_type: str = "gat",
        initial_residual: bool = True,
        representation_separation: bool = True,
        initial_residual_alpha: float = 0.2,
    ):
        super().__init__()
        assert 1 <= num_layers <= 3, "Số lớp GNN phải từ 1 đến 3 (tránh over-smoothing)"

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.use_initial_residual = initial_residual
        self.use_rep_separation = representation_separation

        # ── Representation Separation: hai nhánh encoder độc lập ─────────────
        if representation_separation:
            omics_enc_dim = hidden_channels // 2
            topo_enc_dim  = hidden_channels // 2
            gnn_input_dim = hidden_channels   # concat của hai nhánh
        else:
            omics_enc_dim = hidden_channels
            topo_enc_dim  = 0
            gnn_input_dim = hidden_channels

        self.omics_encoder = OmicsEncoder(omics_dim, omics_enc_dim, dropout=dropout * 0.5)

        if representation_separation and topo_dim > 0:
            self.topo_encoder = TopologyEncoder(topo_dim, topo_enc_dim, dropout=dropout * 0.5)
        else:
            self.topo_encoder = None
            gnn_input_dim = omics_enc_dim

        # ── GNN layers ────────────────────────────────────────────────────────
        self.gnn_layers = nn.ModuleList()
        in_ch = gnn_input_dim
        for _ in range(num_layers):
            self.gnn_layers.append(
                GNNLayer(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    conv_type=conv_type,
                    initial_residual_alpha=initial_residual_alpha if initial_residual else 0.0,
                )
            )
            in_ch = hidden_channels

        # ── Classifier MLP ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),  # Logit (không sigmoid → dùng với FocalLoss)
        )

        self._init_weights()

    def _init_weights(self):
        """Khởi tạo weights theo He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        omics_features: torch.Tensor,
        edge_index: torch.Tensor,
        topo_features: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            omics_features: shape (N, omics_dim) — RNA + Meth + Mut per gene
            edge_index:     shape (2, E) — COO format
            topo_features:  shape (N, topo_dim) — node2vec embeddings (optional)
            edge_weight:    shape (E,) — edge weights (optional)

        Returns:
            logits: shape (N,) — logit cho từng gene (driver probability)
        """
        # ── Encode features ───────────────────────────────────────────────────
        h_omics = self.omics_encoder(omics_features)  # (N, omics_enc_dim)

        if self.use_rep_separation and self.topo_encoder is not None and topo_features is not None:
            h_topo = self.topo_encoder(topo_features)  # (N, topo_enc_dim)
            h = torch.cat([h_omics, h_topo], dim=-1)   # (N, hidden_channels)
        else:
            h = h_omics                                 # (N, hidden_channels)

        # h_0: Lưu initial representation cho Initial Residual
        h0 = h.clone()

        # ── GNN message passing ───────────────────────────────────────────────
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index, x0=h0, edge_weight=edge_weight)

        # ── Classifier ────────────────────────────────────────────────────────
        logits = self.classifier(h).squeeze(-1)  # (N,)
        return logits

    def predict_proba(
        self,
        omics_features: torch.Tensor,
        edge_index: torch.Tensor,
        topo_features: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Trả về xác suất [0,1] thay vì logit."""
        logits = self.forward(omics_features, edge_index, topo_features, edge_weight)
        return torch.sigmoid(logits)

    def get_config(self) -> dict:
        """Trả về config của model để lưu cùng checkpoint."""
        return {
            "num_layers": self.num_layers,
            "hidden_channels": self.hidden_channels,
            "use_initial_residual": self.use_initial_residual,
            "use_rep_separation": self.use_rep_separation,
        }


def build_model_from_config(cfg: dict, omics_dim: int, topo_dim: int) -> MODCANGNNModel:
    """Factory function tạo model từ config.yaml."""
    model_cfg = cfg["model"]
    return MODCANGNNModel(
        omics_dim=omics_dim,
        topo_dim=topo_dim,
        hidden_channels=model_cfg["hidden_channels"],
        num_layers=model_cfg["num_layers"],
        heads=model_cfg["heads"],
        dropout=model_cfg["dropout"],
        conv_type=model_cfg["architecture"],
        initial_residual=model_cfg["initial_residual"],
        representation_separation=model_cfg["representation_separation"],
    )
