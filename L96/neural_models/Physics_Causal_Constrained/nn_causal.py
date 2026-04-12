import torch
import torch.nn as nn

class DiscreteEnergyConservingModel(nn.Module):
    def __init__(self, M_init, F_init, feature_means, feature_stds, N=6, hidden_nodes=100, orthogonal_map='exp'):
        super().__init__()
        self.N = N
        self.num_tri = N * (N - 1) // 2
        self.orthogonal_map = orthogonal_map 
        
        self.register_buffer('mu', torch.tensor(feature_means, dtype=torch.float32))
        self.register_buffer('sigma', torch.tensor(feature_stds, dtype=torch.float32))

        # 1. Initialize Linear Baseline (M_code = M_full - I)
        init_device = M_init.device
        self.M = nn.Parameter(M_init.clone() - torch.eye(N, dtype=torch.float32, device=init_device))
        self.forcing = nn.Parameter(F_init.clone() - (self.mu.to(init_device) / self.sigma.to(init_device)))
        
        # 2. MLP for Nonlinearity
        self.mlp = nn.Sequential(
            nn.Linear(N, hidden_nodes, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_nodes, self.num_tri, bias=False)
        )
        nn.init.zeros_(self.mlp[-1].weight)

        row, col = torch.triu_indices(N, N, offset=1)
        self.register_buffer('row', row)
        self.register_buffer('col', col)

        K = row.shape[0]
        mapping = torch.zeros(K, N * N)
        idx_pos = row * N + col
        idx_neg = col * N + row
        k_idx = torch.arange(K)
        mapping[k_idx, idx_pos] = 1.0
        mapping[k_idx, idx_neg] = -1.0
        self.register_buffer('skew_mapping', mapping)
        self.register_buffer('I', torch.eye(N))

    def _skew_from_triu(self, triu_vals):
        M_flat = triu_vals @ self.skew_mapping
        if triu_vals.dim() == 1:
            return M_flat.view(self.N, self.N)
        else:
            B = triu_vals.shape[0]
            return M_flat.view(B, self.N, self.N)

    def forward(self, x, return_generator=False):
        B = x.shape[0]

        # --- 1. Generate Skew-Symmetric Matrix (The Generator) ---
        S_triu = self.mlp(x) 
        S = self._skew_from_triu(S_triu)
        
        # Physical state for energy-consistent scaling
        u_phys = x * self.sigma + self.mu  

        # ==========================================================
        # FAST PATH: Causal Penalty (Infinitesimal Generator)
        # ==========================================================
        if return_generator:
            # 1. Reconstruct M_full (which is M_code + I)
            M_full = self.M + self.I
            
            # 2. Compute the instantaneous nonlinear push (S * x)
            Su_phys = torch.bmm(S, u_phys.unsqueeze(-1)).squeeze(-1)
            Su_std = Su_phys / self.sigma
            
            # 3. CRITICAL FIX: The linear matrix MIXES the nonlinear push
            # M_full @ (S * x)
            Mixed_Su = Su_std @ M_full.T
            
            # 4. The base linear transition (M_code * x)
            Linear_x = x @ self.M.T  
            
            # Autograd will now correctly evaluate the compositional Jacobian
            return Linear_x + Mixed_Su
        # ==========================================================

        # --- 2. Generate Orthogonal Matrix (Full Rotation) ---
        if self.orthogonal_map == 'cayley':
            I_batch = self.I.unsqueeze(0).expand(B, -1, -1)
            Q = torch.linalg.solve(I_batch - S, I_batch + S)
        elif self.orthogonal_map == 'exp':
            Q = torch.matrix_exp(S)

        # --- 3. Nonlinear Dynamics (Rotation) ---
        Qu_phys = torch.bmm(Q, u_phys.unsqueeze(-1)).squeeze(-1)       
        Qxx = Qu_phys / self.sigma

        # --- 4. Operator Splitting: Linear Dynamics (Scaling) ---
        # M acts on the already-rotated state
        x_rot = Qxx - (self.mu / self.sigma)
        Linear_rot = x_rot @ self.M.T  

        # --- 5. Final Discrete State Update ---
        Fx = self.forcing.unsqueeze(0).expand(B, -1)
        return Fx + Linear_rot + Qxx