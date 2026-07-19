import torch
import math
import numpy as np
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter, scatter_softmax
from e3nn.nn import FullyConnectedNet, Gate
from typing import Dict
from atom_features import create_composite_atom_features
from e3nn.io import CartesianTensor
from stable_loss_implementation import safe_eigh


class EquivariantAttentionPooling(torch.nn.Module):
    """E(3)-equivariant attention pooling with invariant scalar weights."""
    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.score_net = o3.Linear(irreps_in, "1x0e")

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        scores = self.score_net(x)  # [N, 1]
        weights = scatter_softmax(scores, batch, dim=0)
        return scatter(x * weights, batch, dim=0, reduce="sum")


class ShiftedSoftPlus(torch.nn.Module):
    """Shifted softplus: softplus(x) - log(2)"""
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
        self._log2 = math.log(2.0)

    def forward(self, x):
        return self.softplus(x) - self._log2


class EquivariantActivation(torch.nn.Module):
    """Equivariant activation using e3nn's Gate module."""
    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = irreps

        act = {1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates = {1: ShiftedSoftPlus(), -1: torch.tanh}

        scalars_irreps = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0]).simplify()
        gated_irreps = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l > 0])

        if len(scalars_irreps) > 0 and len(gated_irreps) > 0:
            irreps_gates = o3.Irreps([(mul, '0e') for mul, _ in gated_irreps]).simplify()
            scalar_acts = [act[ir.p] for _, ir in scalars_irreps]
            gate_acts = [act_gates[ir.p] for _, ir in irreps_gates]

            self.gate = Gate(
                scalars_irreps, scalar_acts,
                irreps_gates, gate_acts,
                gated_irreps
            )
            self.input_proj = o3.Linear(irreps, scalars_irreps + irreps_gates + gated_irreps)
            self.has_gate = True
            self.irreps_out = self.gate.irreps_out
        else:
            self.has_gate = False
            self.irreps_out = irreps

    def forward(self, x):
        if self.has_gate:
            return self.gate(self.input_proj(x))
        return x


class EquivariantMessagePassing(torch.nn.Module):
    """E(3)-equivariant message passing layer."""
    def __init__(
        self,
        irreps_node_input: o3.Irreps,
        irreps_node_hidden: o3.Irreps,
        irreps_edge_attr: o3.Irreps,
        num_basis: int = 10,
    ):
        super().__init__()
        self.irreps_node_input = irreps_node_input
        self.irreps_node_hidden = irreps_node_hidden
        self.irreps_edge_attr = irreps_edge_attr

        # Tensor product for message passing
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_node_input, irreps_edge_attr, irreps_node_hidden,
            shared_weights=False, internal_weights=False,
        )

        # Radial function network
        self.fc = FullyConnectedNet(
            [num_basis, 32, self.tp.weight_numel],
            act=torch.nn.functional.silu
        )

        # Self-interaction + skip + activation
        self.self_interaction = o3.Linear(irreps_node_hidden, irreps_node_hidden)
        self.skip = o3.Linear(irreps_node_input, irreps_node_hidden)
        self.act = EquivariantActivation(irreps_node_hidden)
        self.irreps_out = self.act.irreps_out

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_length_embedding: torch.Tensor,
        edge_weight: torch.Tensor
    ) -> torch.Tensor:
        weights = self.fc(edge_length_embedding)
        messages = self.tp(node_feats[edge_src], edge_attr, weights) * edge_weight.unsqueeze(-1)

        # Aggregate
        sum_messages = scatter(
            messages,
            edge_dst,
            dim=0,
            dim_size=node_feats.size(0),
            reduce="sum",
        )

        # Normalize by sqrt(count)
        ones = torch.ones(messages.size(0), device=messages.device)
        node_counts = scatter(ones, edge_dst, dim=0, dim_size=node_feats.size(0), reduce="sum")
        agg_messages = sum_messages / torch.sqrt(node_counts + 1e-8).unsqueeze(-1)

        # Self-interaction + skip + activation
        agg_messages = self.self_interaction(agg_messages)
        output = self.act(agg_messages + self.skip(node_feats))

        return output


class EquivariantNetwork(torch.nn.Module):
    """Equivariant network with post-processing block."""
    def __init__(
        self,
        hidden_dim: int = 32,
        max_radius: float = 6.0,
        num_basis: int = 8,
        lmax: int = 2,
        num_layers: int = 2,
        atom_feature_dim: int = 49
    ):
        super().__init__()
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.lmax = lmax
        self.atom_feature_dim = atom_feature_dim

        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        irreps_embedded = f"{atom_feature_dim}x0e"
        irreps_node_hidden = self._build_hidden_irreps(hidden_dim, lmax)

        # Message passing layers
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = EquivariantMessagePassing(
                irreps_embedded if i == 0 else irreps_node_hidden,
                irreps_node_hidden,
                self.irreps_sh,
                num_basis
            )
            if i == 0:
                irreps_embedded = conv.irreps_out
            self.layers.append(conv)
            irreps_node_hidden = conv.irreps_out

        # Post-processing (node-level non-linearity before pooling)
        self.post_linear1 = o3.Linear(irreps_node_hidden, irreps_node_hidden)
        self.post_act = EquivariantActivation(irreps_node_hidden)
        self.post_linear2 = o3.Linear(self.post_act.irreps_out, self.post_act.irreps_out)
        self.irreps_out = self.post_linear2.irreps_out

    def _build_hidden_irreps(self, hidden_dim, lmax):
        """Build hidden irreps with mixed parity."""
        irreps_list = []
        for l in range(lmax + 1):
            mul = max(1, hidden_dim // (2 ** l))
            if l == 0:
                irreps_list.append((mul, (l, 1)))
            else:
                mul_each = max(1, mul // 2)
                irreps_list.append((mul_each, (l, 1)))
                irreps_list.append((mul_each, (l, -1)))

        irreps = o3.Irreps(irreps_list)
        return irreps.simplify()

    def forward(self, data) -> torch.Tensor:
        node_feats = data.node_features

        # Assert feature dimension matches expected
        assert node_feats.shape[-1] == self.atom_feature_dim, \
            f"node_features dim {node_feats.shape[-1]} != expected {self.atom_feature_dim}"

        batch = data.batch
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]
        edge_attr = data.edge_sh
        edge_length_embedding = data.edge_rbf
        edge_weights = data.edge_weights

        # Message passing
        for layer in self.layers:
            node_feats = layer(node_feats, edge_src, edge_dst, edge_attr, edge_length_embedding, edge_weights)

        # Post-processing
        node_feats = self.post_linear1(node_feats)
        node_feats = self.post_act(node_feats)
        node_feats = self.post_linear2(node_feats)

        return node_feats, batch


class EquivariantUncertaintyNetwork(torch.nn.Module):
    """E(3)-equivariant network with uncertainty quantification."""
    def __init__(
        self,
        hidden_dim: int = 128,
        max_radius: float = 6.0,
        atom_feature_dim: int = 49,
        lmax: int = 4,
        num_layers: int = 2,
        covariance_scale: float = 2.0,
        covariance_regularization_weight: float = 0.1,
        # [FIX] Eigenvalue clipping bounds - MUST match loss function parameters
        # These prevent variance collapse (min) and explosion (max) during inference
        min_log_eigenvalue: float = -0.8,  # e^-0.8 ≈ 0.45 min variance (raised from -1.5 to prevent overconfidence)
        max_log_eigenvalue: float = 2.0,   # e^2 ≈ 7.4 max variance
        # [FIX] Detach UQ branch - prevents UQ gradients from affecting feature extractor
        # Set to True if experiencing variance collapse or UQ harming prediction quality
        detach_uq_features: bool = False,
    ):
        super().__init__()
        self.covariance_scale = covariance_scale
        self.min_log_eigenvalue = min_log_eigenvalue
        self.max_log_eigenvalue = max_log_eigenvalue
        self.detach_uq_features = detach_uq_features

        # CartesianTensor for 4th-order covariance tensor (ij=ji, kl=lk, ijkl=klij)
        self.cov_cartesian_tensor = CartesianTensor("ijkl=jikl=ijlk=klij")

        # Feature network
        self.feature_network = EquivariantNetwork(
            hidden_dim=hidden_dim,
            max_radius=max_radius,
            atom_feature_dim=atom_feature_dim,
            lmax=lmax,
            num_layers=num_layers
        )

        # CartesianTensor for symmetric 3x3 tensors
        self.cartesian_tensor = CartesianTensor("ij=ji")

        # Mean branch: attention pooling
        self.mu_pooling = EquivariantAttentionPooling(self.feature_network.irreps_out)
        self.mean_head = o3.Linear(self.feature_network.irreps_out, self.cartesian_tensor)

        # UQ branch: attention pooling + bottleneck
        self.sigma_pooling = EquivariantAttentionPooling(self.feature_network.irreps_out)
        irreps_h = self.feature_network.irreps_out
        self.uq_bottleneck = torch.nn.Sequential(
            o3.Linear(irreps_h, irreps_h),
            EquivariantActivation(irreps_h),
            o3.Linear(irreps_h, irreps_h)
        )
        self.cov_head = o3.Linear(irreps_h, self.cov_cartesian_tensor)

        # Residual A base (isotropic)
        self.A_base = torch.nn.Parameter(torch.ones(1) * 0.0)

        # Register buffers for efficiency
        self.register_buffer("_eye6", torch.eye(6))

        # Pre-compute indices for _cartesian_4d_to_km
        sqrt2 = math.sqrt(2.0)
        km_indices = [(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0), (1, 2, sqrt2), (0, 2, sqrt2), (0, 1, sqrt2)]
        i = torch.tensor([idx[0] for idx in km_indices], dtype=torch.long)
        j = torch.tensor([idx[1] for idx in km_indices], dtype=torch.long)
        f = torch.tensor([idx[2] for idx in km_indices], dtype=torch.float32)
        f_outer = f.unsqueeze(1) * f.unsqueeze(0)

        self.register_buffer("_km_i", i)
        self.register_buffer("_km_j", j)
        self.register_buffer("_km_f", f)
        self.register_buffer("_km_f_outer", f_outer)

        r_idx = torch.arange(6).unsqueeze(1).expand(6, 6)
        c_idx = torch.arange(6).unsqueeze(0).expand(6, 6)
        i1 = i[r_idx]
        j1 = j[r_idx]
        i2 = i[c_idx]
        j2 = j[c_idx]

        self.register_buffer("_km_i1", i1)
        self.register_buffer("_km_j1", j1)
        self.register_buffer("_km_i2", i2)
        self.register_buffer("_km_j2", j2)

    def forward(self, data, compute_sigma: bool = False) -> tuple:
        """Forward pass with dual-branch (mean + uncertainty)."""
        node_feats, batch = self.feature_network(data)

        # Mean branch
        mu_pooled = self.mu_pooling(node_feats, batch)
        mean_irreps = self.mean_head(mu_pooled)
        mu_cartesian = self.cartesian_tensor.to_cartesian(mean_irreps)

        mu_voigt = torch.stack([
            mu_cartesian[:, 0, 0], mu_cartesian[:, 1, 1], mu_cartesian[:, 2, 2],
            mu_cartesian[:, 1, 2], mu_cartesian[:, 0, 2], mu_cartesian[:, 0, 1]
        ], dim=1)

        from voigt_utils import voigt_to_kelvin_mandel
        mu_km = voigt_to_kelvin_mandel(mu_voigt)

        # UQ branch
        # [FIX] Optional detach to prevent UQ gradients from affecting feature_extractor
        # Based on training logs, joint training can cause variance collapse and harm prediction quality
        # Set detach_uq_features=True if:
        #   - LogDet(Sigma) becomes strongly negative (variance collapse)
        #   - Mahalanobis RMS increases while uncertainty shrinks (over-confidence)
        #   - Validation loss increases while MAE plateaus
        sigma_node_input = node_feats.detach() if self.detach_uq_features else node_feats
        sigma_pooled = self.sigma_pooling(sigma_node_input, batch)

        # Residual connection: uq_bottleneck 输出与输入相加
        # 两者具有相同的 irreps (irreps_h)，符合 E(3) 对称性约束
        sigma_h = self.uq_bottleneck(sigma_pooled) + sigma_pooled

        delta_A_irreps = self.cov_head(sigma_h)

        # Convert to KM space
        A_cart = self.cov_cartesian_tensor.to_cartesian(delta_A_irreps)
        delta_A_km = self._cartesian_4d_to_km(A_cart)

        # A = A_base * I + ΔA
        eye = self._eye6.unsqueeze(0).to(delta_A_km.dtype)
        A_km = self.A_base * eye + delta_A_km

        # [FIX] Clean NaN/Inf - use model's eigenvalue bounds instead of hardcoded values
        # posinf=10.0 would cause exp(10) = 22026 explosion! Use max_log_eigenvalue + margin instead
        A_km = torch.nan_to_num(A_km, nan=0.0,
                                posinf=self.max_log_eigenvalue + 0.5,  # ~2.5 instead of 10.0
                                neginf=self.min_log_eigenvalue - 0.5)  # ~-2.0 instead of -10.0

        # Explicit symmetrization: ensures A_km is numerically symmetric
        # This eliminates asymmetric numerical noise from the 4th-order tensor mapping
        # and reduces extreme outliers in D_M^2 tail (99th percentile)
        # Also ensures consistency between training and inference
        A_km = 0.5 * (A_km + A_km.transpose(-2, -1))

        # Note: Removed max_A tanh bounding to allow unconstrained learning
        # Relying on loss function eigenvalue clamping instead

        # Optional: compute Sigma (for inference only)
        Sigma_km = None
        if compute_sigma:
            try:
                A_sym = 0.5 * (A_km + A_km.transpose(-2, -1))
                L, Q = safe_eigh(A_sym.double())

                # [FIX] Use instance parameters instead of hardcoded values
                # MUST match loss function: min_eigenvalue, max_eigenvalue
                L_clamped = torch.clamp(L, min=self.min_log_eigenvalue, max=self.max_log_eigenvalue)

                # Condition number control
                max_log_cond = math.log(200.0)
                eig_min = L_clamped[:, 0]
                eig_max = L_clamped[:, -1]
                current_log_cond = eig_max - eig_min

                exceed_mask = current_log_cond > max_log_cond
                if exceed_mask.any():
                    excess = (current_log_cond - max_log_cond).clamp(min=0.0)
                    eig_max_adjusted = (eig_max - excess).clamp(min=eig_min)
                    L_adjusted = L_clamped.clone()
                    L_adjusted[exceed_mask, -1] = eig_max_adjusted[exceed_mask]
                else:
                    L_adjusted = L_clamped

                Sigma_km = (Q @ torch.diag_embed(torch.exp(L_adjusted)) @ Q.transpose(-2, -1)).float()
            except Exception as e:
                print(f"Warning: Eigenvalue decomposition failed in inference: {e}")
                Sigma_km = torch.eye(6, device=A_km.device).unsqueeze(0).expand_as(A_km)

        return mu_km, A_km, Sigma_km

    def _voigt_to_kelvin_mandel_matrix(self, M_voigt):
        """
        Convert 6x6 matrix from Voigt to Kelvin-Mandel space using proper diagonal scaling.

        Uses congruence transformation with diagonal scaling matrix B:
        M_KM = B @ M_Voigt @ B

        where B = diag(1, 1, 1, √2, √2, √2)
        """
        device, dtype = M_voigt.device, M_voigt.dtype
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))

        # Proper diagonal scaling matrix B
        diag = torch.tensor([1.0, 1.0, 1.0, sqrt2, sqrt2, sqrt2], device=device, dtype=dtype)
        B = torch.diag(diag)

        # Transform: M_KM = B @ M_Voigt @ B
        return B @ M_voigt @ B

    def _kelvin_mandel_to_voigt_matrix(self, M_km):
        """
        Convert 6x6 matrix from Kelvin-Mandel to Voigt space using proper diagonal scaling.

        Uses inverse congruence transformation with diagonal scaling matrix B:
        M_Voigt = B_inv @ M_KM @ B_inv

        where B = diag(1, 1, 1, √2, √2, √2)
        and B_inv = diag(1, 1, 1, 1/√2, 1/√2, 1/√2)
        """
        device, dtype = M_km.device, M_km.dtype
        sqrt2 = torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))

        # Inverse diagonal scaling matrix B_inv
        diag_inv = torch.tensor([1.0, 1.0, 1.0, 1.0/sqrt2, 1.0/sqrt2, 1.0/sqrt2], device=device, dtype=dtype)
        B_inv = torch.diag(diag_inv)

        # Transform: M_Voigt = B_inv @ M_KM @ B_inv
        return B_inv @ M_km @ B_inv

    def predict_with_uncertainty(self, data) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        Always enables compute_sigma=True.

        Returns:
            Dictionary containing:
            - mean: Predicted tensor values (Kelvin-Mandel format)
            - covariance: Full covariance matrix (Kelvin-Mandel)
            - std: Standard deviations
            - correlation: Correlation matrix
        """
        # 强制计算 Sigma
        mu_km, A_km, Sigma_km = self.forward(data, compute_sigma=True)

        # Extract diagonal for standard deviation
        std = torch.sqrt(torch.diagonal(Sigma_km, dim1=-2, dim2=-1))

        # Compute correlation matrix
        std_outer = std.unsqueeze(-1) * std.unsqueeze(-2)
        correlation = Sigma_km / (std_outer + 1e-8)

        return {
            'mean': mu_km,
            'covariance': Sigma_km,
            'std': std,
            'correlation': correlation
        }

    def _cartesian_4d_to_km(self, tensor4d):
        """
        Maps a [B, 3, 3, 3, 3] tensor to [B, 6, 6] Kelvin-Mandel matrix.

        The Kelvin-Mandel representation preserves the Frobenius norm:
            ||T||_F^2 = sum_{km} T_km^2 / factor_k / factor_l

        Vectorized version using pre-computed indices (no temp tensors).
        """
        # Use pre-computed 6x6 index grids (cached as buffers)
        km_matrix = tensor4d[:, self._km_i1, self._km_j1, self._km_i2, self._km_j2]  # [B, 6, 6]

        # Apply factor outer product
        f_outer = self._km_f_outer.to(tensor4d.dtype)  # [6, 6]
        km_matrix = f_outer.unsqueeze(0) * km_matrix  # [B, 6, 6]

        return km_matrix




