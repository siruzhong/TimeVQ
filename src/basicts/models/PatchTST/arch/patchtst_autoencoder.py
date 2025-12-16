"""PatchTST Autoencoder for vector-based prediction."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import RevIN
from ..config.patchtst_config import PatchTSTConfig


class PatchTSTEncoder(nn.Module):
    """Encoder that compresses patches into latent vectors."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.patch_embedding = nn.Linear(config.patch_len, config.hidden_size)
        self.patch_norm = nn.LayerNorm(config.hidden_size)
        
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                MLPLayer(config.hidden_size, config.intermediate_size, hidden_act=config.hidden_act, dropout=config.fc_dropout)
            ) for _ in range(config.num_encoder_layers)
        ])
        
        self.hidden_to_latent = nn.Linear(config.hidden_size, config.latent_size * 2)
        self.norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, patch_len = patches.shape
        patches_flat = patches.reshape(-1, patch_len)
        patch_embeddings_flat = self.patch_norm(self.patch_embedding(patches_flat))
        patch_embeddings = patch_embeddings_flat.reshape(batch_size, num_patches, -1)
        
        hidden_states = patch_embeddings.reshape(-1, patch_embeddings.shape[-1])
        for encoder_layer in self.encoder_layers:
            hidden_states = hidden_states + encoder_layer(hidden_states)
        
        hidden_states = hidden_states.reshape(batch_size, num_patches, -1)
        return self.hidden_to_latent(self.norm(hidden_states))


class PatchTSTDecoder(nn.Module):
    """Decoder that reconstructs patches from latent vectors."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.latent_to_hidden = nn.Linear(config.latent_size, config.hidden_size)
        
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                MLPLayer(config.hidden_size, config.intermediate_size, hidden_act=config.hidden_act, dropout=config.fc_dropout)
            ) for _ in range(config.num_decoder_layers)
        ])
        
        self.hidden_to_patch = nn.Linear(config.hidden_size, config.patch_len)
        self.norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, _ = latent_states.shape
        hidden_states = self.latent_to_hidden(latent_states).reshape(-1, self.latent_to_hidden.out_features)
        
        for decoder_layer in self.decoder_layers:
            hidden_states = hidden_states + decoder_layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        patch_reconstructions = self.hidden_to_patch(hidden_states)
        return patch_reconstructions.reshape(batch_size, num_patches, -1)


class PatchTSTAutoencoder(nn.Module):
    """Complete autoencoder for PatchTST patches."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.encoder = PatchTSTEncoder(config)
        self.decoder = PatchTSTDecoder(config)
        self.ae_dropout = config.ae_dropout
        self.kl_clamp = config.kl_clamp
        self.kl_weight = config.kl_weight
        
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        latent_outputs = self.encoder(patches)
        mean, log_std = torch.chunk(latent_outputs, 2, dim=-1)
        std = torch.exp(log_std.clamp(max=10))
        
        if self.training:
            latent_states = mean + torch.randn_like(mean) * std
            latent_states = F.dropout(latent_states, p=self.ae_dropout, training=True)
        else:
            latent_states = mean
        
        reconstruction = self.decoder(latent_states)
        
        kl_loss = None
        if self.training:
            # KL divergence: -0.5 * sum(1 + log_std^2 - mean^2 - std^2)
            kl_loss = -0.5 * (1 + 2 * log_std - mean.pow(2) - std.pow(2))
            kl_loss = torch.clamp(kl_loss, min=-self.kl_clamp, max=self.kl_clamp)
            kl_loss = kl_loss.sum(dim=-1).mean()
        
        return latent_states, reconstruction, kl_loss


class PatchTSTAutoencoderForForecasting(nn.Module):
    """PatchTST Autoencoder model for time series forecasting."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config
        self.autoencoder = PatchTSTAutoencoder(config)
        
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(
                config.num_features, 
                affine=config.affine, 
                subtract_last=config.subtract_last
            )
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None):
        # Save original inputs for loss calculation (before normalization)
        inputs_original = inputs.clone()
        
        if self.use_revin:
            inputs = self.revin(inputs, "norm")
            
        batch_size, _, num_features = inputs.shape
        patch_len, patch_stride = self.config.patch_len, self.config.patch_stride
        
        # Create patches: [batch_size, num_features, input_len] -> [batch_size * num_features, num_patches, patch_len]
        inputs_t = inputs.transpose(1, 2)
        if self.config.padding:
            inputs_t = nn.ReplicationPad1d((0, patch_stride))(inputs_t)
        patches = inputs_t.unfold(dimension=-1, size=patch_len, step=patch_stride)
        patches = patches.reshape(batch_size * num_features, -1, patch_len)
        
        # Forward through autoencoder
        _, reconstruction, kl_loss = self.autoencoder(patches)
        
        # Reconstruct time series from patches: [batch_size, num_features, num_patches, patch_len] -> [batch_size, seq_len, num_features]
        reconstruction = reconstruction.reshape(batch_size, num_features, -1, patch_len)
        output = reconstruction.permute(0, 2, 3, 1).reshape(batch_size, -1, num_features)
        output = output[:, :self.config.output_len, :]
        
        if self.use_revin:
            output = self.revin(output, "denorm")
        
        # Calculate loss at sequence level (consistent with metrics)
        loss = None
        if self.training and kl_loss is not None:
            seq_recon_loss = F.mse_loss(output, inputs_original)
            loss = seq_recon_loss + self.config.kl_weight * kl_loss
        
        result = {"prediction": output}
        if loss is not None:
            result["loss"] = loss
        return result

