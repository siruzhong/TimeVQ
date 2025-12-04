"""Train PatchTST Autoencoder (VAE) standalone."""
import torch
import torch.nn as nn
from basicts.models.PatchTST import PatchTSTAutoencoder, PatchTSTConfig
from basicts.modules.norm import RevIN
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping, GradientClipping
from basicts import BasicTSLauncher


def patch_mse_metric(prediction=None, targets=None, patch_mse=None, **kwargs):
    """Metric function for patch-level MSE (direct VAE reconstruction quality)."""
    if patch_mse is not None:
        return patch_mse
    return torch.tensor(0.0)


def patch_mae_metric(prediction=None, targets=None, patch_mae=None, **kwargs):
    """Metric function for patch-level MAE (direct VAE reconstruction quality)."""
    if patch_mae is not None:
        return patch_mae
    return torch.tensor(0.0)


class AutoencoderWrapper(nn.Module):
    """Wrapper to adapt autoencoder to BasicTS training pipeline."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config
        self.autoencoder = PatchTSTAutoencoder(config)
        
        self.use_revin = getattr(config, 'use_revin', False)
        if self.use_revin:
            self.revin = RevIN(config.num_features, affine=getattr(config, 'affine', False), subtract_last=getattr(config, 'subtract_last', False))
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None):
        if self.use_revin:
            inputs = self.revin(inputs, "norm")
            
        batch_size, input_len, num_features = inputs.shape
        patch_len, patch_stride = self.config.patch_len, self.config.patch_stride
        
        # Create patches
        inputs_transposed = inputs.transpose(1, 2)
        if self.config.padding:
            inputs_transposed = nn.ReplicationPad1d((0, patch_stride))(inputs_transposed)
        patches = inputs_transposed.unfold(dimension=-1, size=patch_len, step=patch_stride)
        patches = patches.reshape(batch_size * num_features, -1, patch_len) # [batch_size * num_features, num_patches, patch_len]
        
        # Forward through autoencoder
        latent_states, reconstruction, ae_loss = self.autoencoder(patches, targets=patches)
        
        # Calculate patch-level reconstruction error (more direct measure of VAE quality)
        patch_mse = nn.MSELoss()(reconstruction, patches)
        patch_mae = nn.L1Loss()(reconstruction, patches)
        
        # Reconstruct time series (for metrics only, loss is computed at patch level)
        reconstruction_reshaped = reconstruction.reshape(batch_size, num_features, -1, patch_len)
        output_list = []
        for i in range(reconstruction.shape[1]):
            output_list.append(reconstruction_reshaped[:, :, i, :].transpose(1, 2))
        output = torch.cat(output_list, dim=1)[:, :self.config.output_len, :]
        
        result = {"prediction": output}
        if self.training and ae_loss is not None:
            result["loss"] = ae_loss  # BasicTS will use this directly
        # Add patch-level metrics for VAE quality assessment
        result["patch_mse"] = patch_mse
        result["patch_mae"] = patch_mae
        return result


def main():
    model_config = PatchTSTConfig(
        input_len=96, output_len=96, num_features=7,
        patch_len=16, patch_stride=8,
        hidden_size=256, latent_size=16,
        num_encoder_layers=2, num_decoder_layers=2,
        ae_dropout=0.1, kl_weight=1e-3, kl_clamp=0.5,
        num_layers=2, n_heads=1, intermediate_size=256 * 4, fc_dropout=0.1,
        use_revin=True, affine=True, subtract_last=False,
    )
    
    cfg = BasicTSForecastingConfig(
        model=AutoencoderWrapper, model_config=model_config,
        dataset_name="ETTh1", input_len=96, output_len=96,
        gpus="0", num_epochs=100, batch_size=64,
        metrics=["MSE", "MAE", ("PatchMSE", patch_mse_metric), ("PatchMAE", patch_mae_metric)],
        optimizer_params={"lr": 1e-3},
        callbacks=[EarlyStopping(patience=10), GradientClipping(1.0)],
        ckpt_save_dir="checkpoints/PatchTSTAutoencoder", seed=42,
    )
    
    BasicTSLauncher.launch_training(cfg)


if __name__ == "__main__":
    main()
