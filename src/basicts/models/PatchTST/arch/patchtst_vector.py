"""PatchTST_Vector: Vector-based prediction model for PatchTST."""
import torch
import torch.nn as nn
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import RevIN
from basicts.modules.transformer import Encoder, EncoderLayer, MultiHeadAttention
from ..config.patchtst_config import PatchTSTConfig
from .patchtst_autoencoder import PatchTSTAutoencoder
from .patchtst_layers import PatchTSTBatchNorm


class MLPGenerator(nn.Module):
    """Simple MLP generator: latent + noise -> next latent."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.latent_size = config.latent_size
        self.noise_size = config.noise_size
        
        # Simple MLP: concat(latent, noise) -> latent
        self.mlp = nn.Sequential(
            nn.Linear(config.latent_size + config.noise_size, config.latent_size * 2),
            nn.GELU(),
            nn.Linear(config.latent_size * 2, config.latent_size)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
    
    def sample(self, latent_states, deterministic=False):
        """
        Args:
            latent_states: [batch, 1, latent_size]
            deterministic: bool
        Returns:
            [batch, 1, latent_size]
        """
        batch_size = latent_states.shape[0]
        if deterministic:
            noise = torch.zeros((batch_size, 1, self.noise_size), dtype=latent_states.dtype, device=latent_states.device)
        else:
            noise = torch.rand((batch_size, 1, self.noise_size), dtype=latent_states.dtype, device=latent_states.device) - 0.5
        
        # Concat latent and noise, then predict next latent
        x = torch.cat([latent_states, noise], dim=-1)
        return self.mlp(x)


class PatchTSTVectorBackbone(nn.Module):
    """Backbone: patches -> autoencoder.encoder -> Transformer (in latent space)."""
    
    def __init__(self, config: PatchTSTConfig, autoencoder_encoder):
        super().__init__()
        self.autoencoder_encoder = autoencoder_encoder
        self.num_patches = int((config.input_len - config.patch_len) / config.patch_stride + 1) + (1 if config.padding else 0)
        
        norm_type = nn.LayerNorm if config.norm_type == "layer_norm" else PatchTSTBatchNorm
        self.encoder = Encoder(nn.ModuleList([
            EncoderLayer(
                MultiHeadAttention(config.latent_size, config.n_heads, config.attn_dropout),
                MLPLayer(config.latent_size, config.intermediate_size, hidden_act=config.hidden_act, dropout=config.fc_dropout),
                layer_norm=(norm_type, config.latent_size), norm_position="post"
            ) for _ in range(config.num_layers)
        ]))
        self.output_attentions = config.output_attentions
    
    def forward(self, patches):
        """
        Args:
            patches: [batch * num_features, num_patches, patch_len]
        Returns:
            latent_states: [batch * num_features, num_patches, latent_size]
            attn_weights: optional attention weights
        """
        latent_outputs = self.autoencoder_encoder(patches)
        latent_mean, _ = torch.chunk(latent_outputs, 2, dim=-1)
        return self.encoder(latent_mean, output_attentions=self.output_attentions)


class PatchTSTVectorForForecasting(nn.Module):
    """PatchTST_Vector model for time series forecasting using vector-based prediction."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config
        self.patch_len = config.patch_len
        self.patch_stride = config.patch_stride
        self.num_features = config.num_features
        
        self.autoencoder = PatchTSTAutoencoder(config)
        self.freeze_ae = getattr(config, 'freeze_ae', False)
        if self.freeze_ae:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
            self.autoencoder.eval()
        
        self.backbone = PatchTSTVectorBackbone(config, self.autoencoder.encoder)
        self.num_patches = self.backbone.num_patches
        
        self.mlp_generator = MLPGenerator(config)
        
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(config.num_features, affine=config.affine, subtract_last=config.subtract_last)
        
        self.output_attentions = config.output_attentions
        self.num_samples = config.num_samples
        self.beta = config.beta
    
    def _to_patches(self, x):
        """
        Args:
            x: [batch, seq_len, num_features]
        Returns:
            [batch * num_features, num_patches, patch_len]
        """
        x = x.transpose(1, 2)
        if self.config.padding:
            x = nn.ReplicationPad1d((0, self.patch_stride))(x)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return patches.reshape(x.shape[0] * self.num_features, -1, self.patch_len)
    
    def _generate_latents(self, context, num_patches, deterministic=False):
        """
        Args:
            context: [batch * num_features, latent_size]
            num_patches: int
            deterministic: bool
        Returns:
            [batch * num_features, num_patches, latent_size]
        """
        latents = []
        current_latent = context
        for _ in range(num_patches):
            next_latent = self.mlp_generator.sample(current_latent.unsqueeze(1), deterministic=deterministic)
            latents.append(next_latent.squeeze(1))
            current_latent = next_latent.squeeze(1)
        return torch.stack(latents, dim=1)
    
    def energy_score(self, x, mean, log_std):
        """
        Args:
            x: [num_samples, latent_dim]
            mean: [num_patches, latent_dim]
            log_std: [num_patches, latent_dim]
        Returns:
            scalar tensor
        """
        n_x = x.shape[0]
        # distance_x: average pairwise distance within x
        distance_x = torch.pow(torch.linalg.norm(x.unsqueeze(1) - x.unsqueeze(0), ord=2, dim=-1), self.beta)
        distance_x = distance_x.sum(dim=(0, 1)) / (n_x * (n_x - 1))
        
        # distance_y: average distance from x to target distribution samples
        std = torch.exp(log_std.clamp(max=10))
        y = mean + torch.randn((100, *mean.shape), device=mean.device) * std
        distance_y = torch.pow(torch.linalg.norm(x.reshape(n_x, 1, *x.shape[1:]) - y.reshape(1, 100, *y.shape[1:]), ord=2, dim=-1), self.beta)
        distance_y = distance_y.mean(dim=(0, 1))
        return distance_x - distance_y * 2
    
    def forward(self, inputs, targets=None):
        """
        Args:
            inputs: [batch, input_len, num_features]
            targets: [batch, output_len, num_features] or None
        Returns:
            dict with 'prediction' and optionally 'loss', 'attn_weights'
        """
        batch_size = inputs.shape[0]
        
        if self.use_revin:
            inputs = self.revin(inputs, "norm")
            targets_norm = self.revin(targets, "norm") if targets is not None else None
        else:
            targets_norm = targets
        
        input_patches = self._to_patches(inputs)
        latent_states, attn_weights = self.backbone(input_patches)
        last_latent = latent_states[:, -1, :]
        output_patches = (self.config.output_len + self.patch_stride - 1) // self.patch_stride
        
        # Generate predicted latents
        if self.training:
            predicted_latents_list = []
            for _ in range(self.num_samples):
                predicted_latents_list.append(self._generate_latents(last_latent, output_patches, deterministic=False))
            predicted_latents = torch.stack(predicted_latents_list, dim=0)
            predicted_latents_mean = predicted_latents.mean(dim=0)
        else:
            predicted_latents_mean = self._generate_latents(last_latent, output_patches, deterministic=True)
            predicted_latents = predicted_latents_mean.unsqueeze(0).repeat(self.num_samples, 1, 1, 1)
        
        predicted_patches = self.autoencoder.decoder(predicted_latents_mean)
        predicted_patches = predicted_patches.reshape(batch_size, self.num_features, output_patches, self.patch_len)
        prediction = predicted_patches.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_features)
        prediction = prediction[:, :self.config.output_len, :]
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        
        # Compute energy loss
        energy_loss = None
        if targets_norm is not None:
            target_patches = self._to_patches(targets_norm)
            target_patches = target_patches[:, :output_patches, :]
            
            target_latent_outputs = self.autoencoder.encoder(target_patches)
            target_mean, target_log_std = torch.chunk(target_latent_outputs, 2, dim=-1)
            
            predicted_latents_flat = predicted_latents.reshape(self.num_samples, -1, self.config.latent_size)
            target_mean_flat = target_mean.reshape(-1, self.config.latent_size)
            target_log_std_flat = target_log_std.reshape(-1, self.config.latent_size)
            energy_loss = -self.energy_score(predicted_latents_flat, target_mean_flat, target_log_std_flat).mean()
        
        result = {"prediction": prediction}
        if self.output_attentions:
            result["attn_weights"] = attn_weights
        if energy_loss is not None:
            result["loss"] = energy_loss
        return result
