import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.utils.data import DataLoader

# Import your classes from the training script
# Ensure your training script is named 'training.py' in the same folder
from training import DeepBS_Solver, ChronologicalOptionDataset, create_pinn_collate_fn

def analytical_black_scholes(M, tau, r, q, sigma):
    """
    Computes the Black-Scholes formula for a European Call Option.
    Since the network predicts v = C/K (price normalized by strike), 
    we substitute S/K with M (Moneyness).
    """
    # Handle the edge case where tau is very close to 0 to avoid division by zero
    tau = np.maximum(tau, 1e-8)
    
    d1 = (np.log(M) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    
    # C/K = M * e^(-q*tau) * N(d1) - e^(-r*tau) * N(d2)
    v_analytical = M * np.exp(-q * tau) * norm.cdf(d1) - np.exp(-r * tau) * norm.cdf(d2)
    
    # Intrinsic value floor (options can't be worth less than intrinsic value at expiration)
    intrinsic = np.maximum(M - 1.0, 0.0)
    v_analytical = np.where(tau <= 1e-8, intrinsic, v_analytical)
    
    return v_analytical

def plot_test_dataset(model, test_loader, device):
    print("Evaluating Test Dataset...")
    model.eval()
    
    v_true_list = []
    v_hat_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            M, tau, u, r, q, v_true, _ = [b.to(device) for b in batch]
            _, v_hat = model(M, tau, u, r, q)
            
            v_true_list.append(v_true.cpu().numpy())
            v_hat_list.append(v_hat.cpu().numpy())
            
    v_true_all = np.concatenate(v_true_list).flatten()
    v_hat_all = np.concatenate(v_hat_list).flatten()
    
    mse = np.mean((v_true_all - v_hat_all)**2)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(v_true_all, v_hat_all, alpha=0.3, s=5, c='blue', label='Predictions')
    
    # Plot the ideal y=x reference line
    min_val = min(v_true_all.min(), v_hat_all.min())
    max_val = max(v_true_all.max(), v_hat_all.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y=x)')
    
    plt.title(f'Test Dataset: Predicted vs True Normalized Price\nMSE: {mse:.6f}')
    plt.xlabel('True Normalized Price ($v_{true}$)')
    plt.ylabel('Predicted Normalized Price ($v_{hat}$)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('test_predictions_scatter.svg')
    plt.show()
    print("Saved test scatter plot as 'test_predictions_scatter.svg'")

def plot_analytical_sweeps(model, device):
    print("Generating Analytical vs Network 2 Sweeps...")
    model.eval()
    
    # Number of points for the sweep
    N = 100
    
    # Baseline fixed parameters
    M_fixed = 1.0     # At-the-money
    tau_fixed = 1.0   # 1 year to maturity
    r_fixed = 0.05    # 5% risk-free rate
    q_fixed = 0.0     # 0% dividend yield
    sigma_fixed = 0.2 # 20% volatility
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ---------------------------------------------------------
    # Sweep 1: Varying Moneyness (M)
    # ---------------------------------------------------------
    M_sweep = np.linspace(0.5, 1.5, N).astype(np.float32)
    
    inputs_1 = torch.tensor(np.column_stack([
        M_sweep, 
        np.full(N, tau_fixed), 
        np.full(N, r_fixed), 
        np.full(N, q_fixed), 
        np.full(N, sigma_fixed)
    ]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        v_hat_1 = model.nn2(inputs_1).cpu().numpy().flatten()
        
    v_true_1 = analytical_black_scholes(M_sweep, tau_fixed, r_fixed, q_fixed, sigma_fixed)
    
    axes[0].plot(M_sweep, v_true_1, 'k-', lw=2, label='Analytical BS')
    axes[0].scatter(M_sweep, v_hat_1, c='red', s=15, label='PINN_MLP Predict', zorder=3)
    axes[0].set_title(f'Varying Moneyness (M)\n$\\tau$={tau_fixed}, r={r_fixed}, $\\sigma$={sigma_fixed}')
    axes[0].set_xlabel('Moneyness (M = S/K)')
    axes[0].set_ylabel('Normalized Price (v = C/K)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # ---------------------------------------------------------
    # Sweep 2: Varying Time to Maturity (tau)
    # ---------------------------------------------------------
    tau_sweep = np.linspace(0.01, 3.0, N).astype(np.float32)
    
    inputs_2 = torch.tensor(np.column_stack([
        np.full(N, M_fixed), 
        tau_sweep, 
        np.full(N, r_fixed), 
        np.full(N, q_fixed), 
        np.full(N, sigma_fixed)
    ]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        v_hat_2 = model.nn2(inputs_2).cpu().numpy().flatten()
        
    v_true_2 = analytical_black_scholes(M_fixed, tau_sweep, r_fixed, q_fixed, sigma_fixed)
    
    axes[1].plot(tau_sweep, v_true_2, 'k-', lw=2, label='Analytical BS')
    axes[1].scatter(tau_sweep, v_hat_2, c='red', s=15, label='PINN_MLP Predict', zorder=3)
    axes[1].set_title(f'Varying Time ($\\tau$)\nM={M_fixed}, r={r_fixed}, $\\sigma$={sigma_fixed}')
    axes[1].set_xlabel('Time to Maturity in Years ($\\tau$)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    # ---------------------------------------------------------
    # Sweep 3: Varying Volatility (sigma)
    # ---------------------------------------------------------
    sigma_sweep = np.linspace(0.05, 0.8, N).astype(np.float32)
    
    inputs_3 = torch.tensor(np.column_stack([
        np.full(N, M_fixed), 
        np.full(N, tau_fixed), 
        np.full(N, r_fixed), 
        np.full(N, q_fixed), 
        sigma_sweep
    ]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        v_hat_3 = model.nn2(inputs_3).cpu().numpy().flatten()
        
    v_true_3 = analytical_black_scholes(M_fixed, tau_fixed, r_fixed, q_fixed, sigma_sweep)
    
    axes[2].plot(sigma_sweep, v_true_3, 'k-', lw=2, label='Analytical BS')
    axes[2].scatter(sigma_sweep, v_hat_3, c='red', s=15, label='PINN_MLP Predict', zorder=3)
    axes[2].set_title(f'Varying Volatility ($\\sigma$)\nM={M_fixed}, $\\tau$={tau_fixed}, r={r_fixed}')
    axes[2].set_xlabel('Volatility ($\\sigma$)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig('analytical_sweeps.svg')
    plt.show()
    print("Saved analytical sweeps plot as 'analytical_sweeps.svg'")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize model architecture (must match training parameters)
    model = DeepBS_Solver(hidden_dim=64, num_layers=4, p_dim=64).to(device)
    
    # 2. Load the trained weights
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        print("Successfully loaded 'best_model.pth'")
    except FileNotFoundError:
        print("Error: 'best_model.pth' not found. Please ensure the path is correct.")
        return

    # 3. Load the test dataset for the scatter plot
    print("Loading test dataset...")
    test_ds = ChronologicalOptionDataset('../../wrds_data/deeponet_tensors.h5', 'test')
    test_loader = DataLoader(
        test_ds, 
        batch_size=512, 
        shuffle=False,
        collate_fn=create_pinn_collate_fn(colloc_ratio=0.0)
    )
    
    # 4. Execute the plots
    plot_test_dataset(model, test_loader, device)
    plot_analytical_sweeps(model, device)

if __name__ == "__main__":
    main()