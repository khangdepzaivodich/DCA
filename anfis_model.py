"""
ANFIS: Adaptive Network-based Fuzzy Inference System (PyTorch)
Implements the exact 5-layer architecture from doc3.md (Jang, 1993):

  Layer 1 - Fuzzification:   Gaussian membership functions μ(x) = exp(-((x-c)/a)^2)
  Layer 2 - Rule Layer:      Firing strengths w_i = product of μ across inputs  
  Layer 3 - Normalization:   w_bar_i = w_i / sum(w_i)
  Layer 4 - Defuzzification: Sugeno linear output f_i = p*x1 + q*x2 + ... + r
  Layer 5 - Aggregation:     output = sum(w_bar_i * f_i)

Inputs:  4 wavelet components (A3, D3, D2, D1)
Output:  Predicted price (scalar)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans


class GaussianMembership(nn.Module):
    """
    Layer 1: Fuzzification Layer
    μ_A(x) = exp( -((x - c) / a)^2 )
    Learnable premise parameters: {a (width), c (center)}
    """
    def __init__(self, n_inputs, n_rules):
        super().__init__()
        # Centers (c) and widths (a) for each input × each rule
        self.centers = nn.Parameter(torch.randn(n_inputs, n_rules))  
        self.widths = nn.Parameter(torch.ones(n_inputs, n_rules))    
    
    def forward(self, x):
        """
        x: (batch, n_inputs)
        returns: (batch, n_inputs, n_rules) membership values
        """
        # x: (batch, n_inputs) -> (batch, n_inputs, 1)
        x = x.unsqueeze(2)
        # Gaussian: exp(-((x - c) / a)^2)
        # Clamp widths to prevent division by zero
        a = torch.clamp(self.widths.abs(), min=1e-6)
        mu = torch.exp(-((x - self.centers) / a) ** 2)
        return mu


class ANFIS(nn.Module):
    """
    Full 5-layer ANFIS architecture (Sugeno-type first-order).
    
    Args:
        n_inputs:  Number of input features (4 for wavelet: A3, D3, D2, D1)
        n_rules:   Number of fuzzy rules (clusters from FCM/KMeans)
    """
    def __init__(self, n_inputs=4, n_rules=8):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        
        # Layer 1: Gaussian Membership Functions
        self.membership = GaussianMembership(n_inputs, n_rules)
        
        # Layer 4: Consequent (Sugeno) parameters
        # For each rule: f_i = p_i1*x1 + p_i2*x2 + ... + p_in*xn + r_i
        # Shape: (n_rules, n_inputs + 1) — the +1 is the bias term r_i
        self.consequent_weights = nn.Parameter(torch.randn(n_rules, n_inputs))
        self.consequent_bias = nn.Parameter(torch.zeros(n_rules))
    
    def forward(self, x):
        """
        x: (batch, n_inputs)
        returns: (batch, 1) predicted output
        """
        
        # ---- Layer 1: Fuzzification ----
        # mu: (batch, n_inputs, n_rules)
        mu = self.membership(x)
        
        # ---- Layer 2: Rule Firing Strengths ----
        # w_i = product of mu across inputs for each rule
        # w: (batch, n_rules)
        w = torch.prod(mu, dim=1)
        
        # ---- Layer 3: Normalization ----
        # w_bar_i = w_i / sum(w_j)
        w_sum = w.sum(dim=1, keepdim=True) + 1e-8
        w_bar = w / w_sum  # (batch, n_rules)
        
        # ---- Layer 4: Defuzzification (Sugeno first-order) ----
        # f_i = sum(p_ij * x_j) + r_i for each rule
        # x: (batch, n_inputs) @ consequent_weights.T: (n_inputs, n_rules) = (batch, n_rules)
        f = torch.matmul(x, self.consequent_weights.T) + self.consequent_bias  # (batch, n_rules)
        
        # Weighted output: w_bar_i * f_i
        weighted = w_bar * f  # (batch, n_rules)
        
        # ---- Layer 5: Aggregation ----
        output = weighted.sum(dim=1, keepdim=True)  # (batch, 1)
        
        return output
    
    def initialize_from_data(self, X_train, y_train=None):
        """Standard KMeans Initialization"""
        if hasattr(X_train, 'cpu'):
            X_train_np = X_train.detach().cpu().numpy()
        else:
            X_train_np = X_train

        km = KMeans(n_clusters=self.n_rules, random_state=42, n_init=10)
        km.fit(X_train_np)
        
        centers = torch.tensor(km.cluster_centers_.T, dtype=torch.float32).to(X_train.device)
        self.membership.centers.data = centers
        self.membership.widths.data.fill_(X_train_np.std())
        
        nn.init.xavier_uniform_(self.consequent_weights)
        nn.init.zeros_(self.consequent_bias)

    def initialize_expert_priors(self):
        """
        V10: Institutional Seeding.
        Forces rule centers to align with 'Common Sense' market states.
        """
        with torch.no_grad():
            # Centers shape: [n_inputs, n_rules]
            # Filling with 0.3, 0.5, 0.7 patterns
            for r in range(self.n_rules):
                val = 0.3 if r < 4 else (0.5 if r < 8 else 0.7)
                self.membership.centers.data[:, r].fill_(val)
            
            self.membership.widths.data.fill_(0.15)
            
        nn.init.xavier_uniform_(self.consequent_weights)
        nn.init.zeros_(self.consequent_bias)
        print(f"[ANFIS] Expert Priors Initialized (n_rules={self.n_rules})")


def train_anfis(model, X_train, y_train, X_val=None, y_val=None,
                epochs=300, lr=0.01, batch_size=64, device='cuda'):
    """
    Train ANFIS using hybrid learning:
    - Premise params (membership a, c): gradient descent (Adam)
    - Consequent params (p, q, r): gradient descent (Adam)
    
    The paper uses hybrid backprop+LSE, but Adam on all params works well in practice.
    """
    model = model.to(device)
    
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    
    if X_val is not None:
        X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_v = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        
        epoch_loss /= len(X_t)
        
        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = criterion(val_pred, y_v).item()
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 50 == 0:
                val_mae = torch.mean(torch.abs(val_pred - y_v)).item()
                print(f"  Epoch {epoch+1:3d} | Train MSE: {epoch_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f}")
            
            # Early stopping
            if patience_counter >= 40:
                print(f"  Early stop at epoch {epoch+1}")
                break
        else:
            scheduler.step(epoch_loss)
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1:3d} | Train MSE: {epoch_loss:.2f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model


if __name__ == "__main__":
    # Quick sanity test
    model = ANFIS(n_inputs=4, n_rules=8)
    x = torch.randn(16, 4)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Sample output: {out[:3].detach().numpy().flatten()}")
    print(f"Total params: {sum(p.numel() for p in model.parameters())}")
