"""Simple visualization script for Autoencoder reconstruction."""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse

# Dataset to num_features mapping
DATASET_NUM_FEATURES = {
    "ETTh1": 7,
    "ETTh2": 7,
    "ETTm1": 7,
    "ETTm2": 7,
    "Electricity": 321,
    "Weather": 21,
    "ExchangeRate": 8,
}


def visualize(checkpoint_base: str, dataset_name: str, seq_len: int = 96):
    """Visualize autoencoder reconstruction results.
    
    Args:
        checkpoint_base: Base directory for checkpoints (e.g., "checkpoints/PatchTSTAutoencoder")
        dataset_name: Name of the dataset (e.g., "ExchangeRate")
        seq_len: Sequence length (default: 96)
    """
    # Get num_features from mapping
    if dataset_name not in DATASET_NUM_FEATURES:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_NUM_FEATURES.keys())}")
    num_features = DATASET_NUM_FEATURES[dataset_name]
    
    # Find latest checkpoint
    checkpoint_dir = os.path.join(checkpoint_base, dataset_name)
    checkpoint_dirs = glob.glob(os.path.join(checkpoint_dir, "*/test_results"))
    if not checkpoint_dirs:
        print(f"No test_results found in {checkpoint_dir}")
        return
    
    checkpoint_dirs.sort(key=os.path.getmtime, reverse=True)
    result_dir = checkpoint_dirs[0]
    print(f"Loading from: {result_dir}")
    
    # Load data
    inputs = np.memmap(os.path.join(result_dir, "inputs.npy"), mode='r', dtype='float32')
    prediction = np.memmap(os.path.join(result_dir, "prediction.npy"), mode='r', dtype='float32')
    
    # Reshape: (num_samples, seq_len, num_features)
    num_samples = inputs.size // (seq_len * num_features)
    inputs = np.array(inputs.reshape(num_samples, seq_len, num_features))
    prediction = np.array(prediction.reshape(num_samples, seq_len, num_features))
    
    print(f"Loaded {num_samples} samples")
    print(f"Inputs shape: {inputs.shape}, Prediction shape: {prediction.shape}")
    
    # Calculate metrics
    mse = np.mean((inputs - prediction) ** 2)
    mae = np.mean(np.abs(inputs - prediction))
    print(f"\nOverall MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    # Visualize first 5 samples
    num_samples_viz = min(5, num_samples)
    num_features = inputs.shape[2]
    
    fig, axes = plt.subplots(num_samples_viz, num_features, 
                            figsize=(4*num_features, 3*num_samples_viz))
    if num_samples_viz == 1:
        axes = axes[np.newaxis, :]
    if num_features == 1:
        axes = axes[:, np.newaxis]
    
    for i in range(num_samples_viz):
        for j in range(num_features):
            ax = axes[i, j]
            ax.plot(inputs[i, :, j], label='Input', linewidth=2, alpha=0.8, color='blue')
            ax.plot(prediction[i, :, j], label='Reconstruction', linewidth=2, linestyle='--', alpha=0.8, color='orange')
            
            mse_feat = np.mean((inputs[i, :, j] - prediction[i, :, j]) ** 2)
            mae_feat = np.mean(np.abs(inputs[i, :, j] - prediction[i, :, j]))
            ax.set_title(f'Sample {i+1}, Feature {j+1}\nMSE: {mse_feat:.4f}, MAE: {mae_feat:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(result_dir, "reconstruction.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize autoencoder reconstruction results")
    parser.add_argument("--checkpoint_base", type=str, default="checkpoints/PatchTSTAutoencoder", help="Base directory for checkpoints")
    parser.add_argument("--dataset", type=str, default="ExchangeRate", choices=list(DATASET_NUM_FEATURES.keys()), help="Dataset name")
    parser.add_argument("--seq_len", type=int, default=96, help="Sequence length (default: 96)")
    
    args = parser.parse_args()
    
    visualize(
        checkpoint_base=args.checkpoint_base,
        dataset_name=args.dataset,
        seq_len=args.seq_len
    )

# python visualize_ae.py --checkpoint_base checkpoints/PatchTSTAutoencoder --dataset ExchangeRate --seq_len 96
if __name__ == "__main__":
    main()

