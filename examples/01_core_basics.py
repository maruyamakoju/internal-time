"""Basic usage of TemporalGRUCell — the core building block.

Shows how to:
1. Create a TemporalGRUCell
2. Process a single step
3. Process a full sequence
4. Inspect internal time signals (delta_tau, pred_error, alpha)
"""

import torch
from internal_time import TemporalGRUCell

# --- 1. Create the cell ---
cell = TemporalGRUCell(
    input_dim=5,       # e.g. 5 sensor channels
    hidden_dim=32,     # GRU hidden size
    use_self_model=True,
)
print(f"Model: {cell}")
print(f"Parameters: {sum(p.numel() for p in cell.parameters()):,}")

# --- 2. Single step ---
batch_size = 4
h = cell.init_hidden(batch_size)
x_t = torch.randn(batch_size, 5)

step_out = cell(x_t, h)
print(f"\nSingle step:")
print(f"  hidden shape:  {step_out.hidden.shape}")
print(f"  delta_tau:     {step_out.delta_tau}")
print(f"  alpha:         {step_out.alpha}")
print(f"  pred_error:    {step_out.pred_error}")

# --- 3. Process a sequence ---
seq_len = 100
x_seq = torch.randn(batch_size, seq_len, 5)

seq_out = cell.forward_sequence(x_seq)
print(f"\nSequence output:")
print(f"  hiddens shape:    {seq_out.hiddens.shape}")
print(f"  delta_taus shape: {seq_out.delta_taus.shape}")
print(f"  self_model_loss:  {seq_out.self_model_loss.item():.6f}")

# --- 4. Inspect temporal dynamics ---
# delta_tau shows where the model "pays attention" in time
tau = seq_out.delta_taus[0].detach().numpy()
pe = seq_out.pred_errors[0].detach().numpy()

print(f"\nTemporal dynamics (first sample):")
print(f"  delta_tau mean: {tau.mean():.4f}, std: {tau.std():.4f}")
print(f"  pred_error mean: {pe.mean():.4f}, std: {pe.std():.4f}")
print(f"  tau range: [{tau.min():.4f}, {tau.max():.4f}]")

# --- 5. Use in a training loop (sketch) ---
optimizer = torch.optim.Adam(cell.parameters(), lr=1e-3)

for epoch in range(3):
    out = cell.forward_sequence(x_seq)
    loss = out.self_model_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  epoch {epoch + 1}: self_model_loss = {loss.item():.6f}")

print("\nDone!")
