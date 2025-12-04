"""PatchTST Autoencoder for vector-based prediction."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicts.modules.mlps import MLPLayer
from ..config.patchtst_config import PatchTSTConfig


class PatchTSTEncoder(nn.Module):
    """Encoder that compresses patches into latent vectors."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.patch_len = config.patch_len
        self.latent_size = config.latent_size if hasattr(config, 'latent_size') else config.hidden_size
        self.hidden_size = config.hidden_size
        
        self.patch_embedding = nn.Linear(config.patch_len, config.hidden_size)
        self.patch_norm = nn.LayerNorm(config.hidden_size)
        
        num_encoder_layers = config.num_encoder_layers if hasattr(config, 'num_encoder_layers') else 2
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                MLPLayer(config.hidden_size, config.intermediate_size, hidden_act=config.hidden_act, dropout=config.fc_dropout)
            ) for _ in range(num_encoder_layers)
        ])
        
        self.hidden_to_latent = nn.Linear(config.hidden_size, self.latent_size * 2)
        self.norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if len(patches.shape) == 2:
            patches = patches.unsqueeze(1)
        
        batch_size, num_patches, patch_len = patches.shape
        patches_flat = patches.reshape(-1, patch_len)   # [batch_size * num_patches, patch_len]
        patch_embeddings_flat = self.patch_norm(self.patch_embedding(patches_flat))
        patch_embeddings = patch_embeddings_flat.reshape(batch_size, num_patches, self.hidden_size)
        
        hidden_states = patch_embeddings.reshape(-1, self.hidden_size)  # [batch_size * num_patches, hidden_size]
        for encoder_layer in self.encoder_layers:
            hidden_states = hidden_states + encoder_layer(hidden_states)
        
        hidden_states = hidden_states.reshape(batch_size, num_patches, self.hidden_size)
        return self.hidden_to_latent(self.norm(hidden_states))  # [batch_size, num_patches, latent_size * 2]


class PatchTSTDecoder(nn.Module):
    """Decoder that reconstructs patches from latent vectors."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.patch_len = config.patch_len
        self.latent_size = config.latent_size if hasattr(config, 'latent_size') else config.hidden_size
        self.hidden_size = config.hidden_size
        
        self.latent_to_hidden = nn.Linear(self.latent_size, config.hidden_size)
        
        num_decoder_layers = config.num_decoder_layers if hasattr(config, 'num_decoder_layers') else 2
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                MLPLayer(config.hidden_size, config.intermediate_size, hidden_act=config.hidden_act, dropout=config.fc_dropout)
            ) for _ in range(num_decoder_layers)
        ])
        
        self.hidden_to_patch = nn.Linear(config.hidden_size, config.patch_len)
        self.norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, latent_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, latent_size = latent_states.shape
        hidden_states = self.latent_to_hidden(latent_states).reshape(-1, self.hidden_size)  # [batch_size * num_patches, hidden_size]
        
        for decoder_layer in self.decoder_layers:
            hidden_states = hidden_states + decoder_layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)  # [batch_size * num_patches, hidden_size]
        patch_reconstructions = self.hidden_to_patch(hidden_states)  # [batch_size * num_patches, patch_len]
        return patch_reconstructions.reshape(batch_size, num_patches, self.patch_len)  # [batch_size, num_patches, patch_len]


class PatchTSTAutoencoder(nn.Module):
    """Complete autoencoder for PatchTST patches."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.encoder = PatchTSTEncoder(config)
        self.decoder = PatchTSTDecoder(config)
        self.patch_len = config.patch_len
        self.latent_size = config.latent_size if hasattr(config, 'latent_size') else config.hidden_size
        self.ae_dropout = config.ae_dropout if hasattr(config, 'ae_dropout') else 0.15
        self.kl_clamp = config.kl_clamp if hasattr(config, 'kl_clamp') else 0.5
        self.kl_weight = config.kl_weight if hasattr(config, 'kl_weight') else 1e-3
        
    def forward(self, patches: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        latent_outputs = self.encoder(patches)
        mean, log_std = torch.chunk(latent_outputs, 2, dim=-1)
        std = torch.exp(log_std.clamp(max=10))
        
        if self.training:
            latent_states = mean + torch.randn_like(mean) * std
            latent_states = F.dropout(latent_states, p=self.ae_dropout, training=self.training)
        else:
            latent_states = mean
        
        reconstruction = self.decoder(latent_states)
        
        loss = None
        if targets is not None and self.training:
            recon_loss = F.mse_loss(reconstruction, targets, reduction='mean')
            kl_loss = 0.5 * (torch.pow(mean, 2) + torch.pow(std, 2) - 1 - log_std * 2)
            kl_loss = torch.clamp(kl_loss, min=-self.kl_clamp, max=self.kl_clamp)
            kl_loss = torch.mean(torch.sum(kl_loss, dim=-1))
            loss = recon_loss + self.kl_weight * kl_loss
        
        return latent_states, reconstruction, loss

