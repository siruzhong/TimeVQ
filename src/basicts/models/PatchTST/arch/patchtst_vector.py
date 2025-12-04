"""PatchTST_Vector: Vector-based prediction model for PatchTST."""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from basicts.modules.decomposition import MovingAverageDecomposition
from basicts.modules.embed import PatchEmbedding
from basicts.modules.mlps import MLPLayer
from basicts.modules.norm import RevIN
from basicts.modules.transformer import Encoder, EncoderLayer, MultiHeadAttention
from ..config.patchtst_config import PatchTSTConfig
from .patchtst_autoencoder import PatchTSTAutoencoder
from .patchtst_layers import PatchTSTBatchNorm


class MLPBlock(nn.Module):
    """Residual block for MLP-based generative head."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(2 * channels, channels, bias=True), nn.GELU(),
            nn.Linear(channels, channels, bias=True), nn.GELU(),
            nn.Linear(channels, 2 * channels, bias=True)
        )
        self.gate_act = nn.GELU()
        self.down_proj = nn.Linear(channels, channels, bias=True)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.linears(torch.cat((self.in_ln(x), y), dim=-1))
        gate_proj, up_proj = torch.chunk(h, 2, dim=-1)
        return x + self.down_proj(self.gate_act(gate_proj) * up_proj)


class FinalLayer(nn.Module):
    """Final projection layer for MLP generator."""
    
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(model_channels, eps=1e-6)
        self.linears = nn.Sequential(
            nn.Linear(model_channels, model_channels, bias=True), nn.GELU(),
            nn.Linear(model_channels, out_channels, bias=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linears(self.in_ln(x))


class MLPGenerator(nn.Module):
    """MLP-based generative head for predicting latent vectors."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.noise_size = config.noise_size
        self.latent_size = config.latent_size
        self.hidden_size = config.hidden_size
        
        self.noise_embd = nn.Linear(config.noise_size, config.hidden_size)
        self.hidden_embd = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm_hidden = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm_noise = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        self.mlp_blocks = nn.ModuleList([MLPBlock(config.hidden_size) for _ in range(config.num_mlp_layers)])
        self.final_layer = FinalLayer(config.hidden_size, config.latent_size)
        
        nn.init.constant_(self.final_layer.linears[-1].weight, 0)
        nn.init.constant_(self.final_layer.linears[-1].bias, 0)
    
    def sample(self, hidden_states: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Prepare noise
        if deterministic:
             noise = torch.zeros(
                (*hidden_states.shape[:-1], self.noise_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
        else:
            noise = torch.rand(
                (*hidden_states.shape[:-1], self.noise_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            ) - 0.5
        
        # Embed and normalize
        noise_embds = self.norm_noise(self.noise_embd(noise))
        hidden_states = self.norm_hidden(self.hidden_embd(hidden_states))
        for block in self.mlp_blocks:
            noise_embds = block(noise_embds, hidden_states)
        return self.final_layer(noise_embds)


class PatchTSTVectorBackbone(nn.Module):
    """Backbone for PatchTST_Vector that processes patches and generates hidden states."""
    
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.num_features = config.num_features
        
        # Patching and embedding
        padding = (0, config.patch_stride) if config.padding else None
        self.patch_embedding = PatchEmbedding(config.hidden_size, config.patch_len, config.patch_stride, padding, config.fc_dropout)
        self.num_patches = int((config.input_len - config.patch_len) / config.patch_stride + 1) + (1 if config.padding else 0)
        
        norm_type = nn.LayerNorm if config.norm_type == "layer_norm" else PatchTSTBatchNorm
        self.encoder = Encoder(nn.ModuleList([
            EncoderLayer(
                MultiHeadAttention(config.hidden_size, config.n_heads, config.attn_dropout),
                MLPLayer(config.hidden_size, config.intermediate_size, hidden_act=config.hidden_act, dropout=config.fc_dropout),
                layer_norm=(norm_type, config.hidden_size), norm_position="post"
            ) for _ in range(config.num_layers)
        ]))
        self.output_attentions = config.output_attentions
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Optional[list]]:
        hidden_states = self.patch_embedding(inputs)
        return self.encoder(hidden_states, output_attentions=self.output_attentions)


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
        
        self.decomp = config.decomp
        if self.decomp:
            self.decomp_layer = MovingAverageDecomposition(config.moving_avg)
            self.seasonal_backbone = PatchTSTVectorBackbone(config)
            self.trend_backbone = PatchTSTVectorBackbone(config)
            self.num_patches = self.seasonal_backbone.num_patches
        else:
            self.backbone = PatchTSTVectorBackbone(config)
            self.num_patches = self.backbone.num_patches
        
        self.latent_to_hidden_proj = nn.Sequential(
            nn.Linear(config.latent_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=1e-6)
        )
        self.mlp_generator = MLPGenerator(config)
        
        self.use_revin = config.use_revin
        if self.use_revin:
            self.revin = RevIN(config.num_features, affine=config.affine, subtract_last=config.subtract_last)
        
        self.output_attentions = config.output_attentions
        self.num_samples = config.num_samples
        self.beta = config.beta
    
    def distance(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.linalg.norm(x_1 - x_2, ord=2, dim=-1), self.beta)
    
    def energy_score(self, x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        n_x = x.shape[0]
        distance_matrix = self.distance(x.unsqueeze(1), x.unsqueeze(0))
        distance_x = distance_matrix.sum(dim=(0, 1)) / (n_x * (n_x - 1))
        
        std = torch.exp(log_std.clamp(max=10))
        n_y = 100
        y = mean + torch.randn((n_y, *mean.shape), device=mean.device) * std
        
        distance_y = self.distance(x.reshape(n_x, 1, *x.shape[1:]), y.reshape(1, n_y, *y.shape[1:])).mean(dim=(0, 1))
        return distance_x - distance_y * 2
    
    def forward(self, inputs: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = inputs.shape[0]
        
        if self.use_revin:
            inputs = self.revin(inputs, "norm")
            targets_norm = self.revin(targets, "norm") if targets is not None else None
        else:
            targets_norm = targets
        
        if self.decomp:
            seasonal_hidden, attn_weights = self.seasonal_backbone(inputs)
            patch_embeddings = seasonal_hidden + self.trend_backbone(inputs)[0]
        else:
            patch_embeddings, attn_weights = self.backbone(inputs)
        
        last_patch_embedding = patch_embeddings[:, -1, :]
        output_patches = (self.config.output_len + self.patch_stride - 1) // self.patch_stride
        
        if self.training and targets_norm is not None:
            # Training branch: compute energy loss
            targets_transposed = targets_norm.transpose(1, 2)
            if self.config.padding:
                targets_transposed = nn.ReplicationPad1d((0, self.patch_stride))(targets_transposed)
            
            target_patches = targets_transposed.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
            target_patches = target_patches.reshape(batch_size * self.num_features, -1, self.patch_len)
            
            # Encode target patches to get target distribution
            was_training = self.autoencoder.training
            if self.freeze_ae:
                self.autoencoder.eval()
            
            with torch.set_grad_enabled(not self.freeze_ae):
                target_latent_outputs = self.autoencoder.encoder(target_patches)
                target_mean, target_log_std = torch.chunk(target_latent_outputs, 2, dim=-1)
            
            if not self.freeze_ae and was_training:
                self.autoencoder.train()
            
            # Generate multiple samples for energy score computation
            last_patch_embedding_repeated = last_patch_embedding.unsqueeze(0).repeat(self.num_samples, 1, 1)
            predicted_latents_list = []
            for sample_idx in range(self.num_samples):
                current_context = last_patch_embedding_repeated[sample_idx]
                sample_latents = []
                for _ in range(output_patches):
                    next_latent = self.mlp_generator.sample(current_context.unsqueeze(1))
                    sample_latents.append(next_latent.squeeze(1))
                    current_context = self.latent_to_hidden_proj(next_latent.squeeze(1))
                predicted_latents_list.append(torch.stack(sample_latents, dim=1))
            
            predicted_latents = torch.stack(predicted_latents_list, dim=0)
            predicted_latents_flat = predicted_latents.reshape(self.num_samples, -1, self.config.latent_size)
            target_mean_flat = target_mean.reshape(-1, self.config.latent_size)
            target_log_std_flat = target_log_std.reshape(-1, self.config.latent_size)
            
            energy_loss = -self.energy_score(predicted_latents_flat, target_mean_flat, target_log_std_flat).mean()
            predicted_latents_mean = predicted_latents.mean(dim=0)
        else:
            # Inference branch: deterministic generation
            predicted_latents_mean = []
            current_context = last_patch_embedding
            for _ in range(output_patches):
                next_latent = self.mlp_generator.sample(current_context.unsqueeze(1), deterministic=not self.training)
                predicted_latents_mean.append(next_latent.squeeze(1))
                current_context = self.latent_to_hidden_proj(next_latent.squeeze(1))
            predicted_latents_mean = torch.stack(predicted_latents_mean, dim=1)
            energy_loss = None
        
        # Decode predicted latents to patches
        with torch.set_grad_enabled(not self.freeze_ae):
            predicted_patches = self.autoencoder.decoder(predicted_latents_mean)
        
        predicted_patches = predicted_patches.reshape(batch_size, self.num_features, output_patches, self.patch_len)
        
        if self.patch_stride < self.patch_len:
            actual_output_len = (output_patches - 1) * self.patch_stride + self.patch_len
            prediction = torch.zeros(batch_size, actual_output_len, self.num_features, device=predicted_patches.device, dtype=predicted_patches.dtype)
            count = torch.zeros(batch_size, actual_output_len, self.num_features, device=predicted_patches.device, dtype=torch.float32)
            for i in range(output_patches):
                start_idx = i * self.patch_stride
                patch = predicted_patches[:, :, i, :].transpose(1, 2)
                prediction[:, start_idx:start_idx + self.patch_len, :] += patch
                count[:, start_idx:start_idx + self.patch_len, :] += 1.0
            prediction = prediction / count.clamp(min=1.0)
        else:
            prediction = torch.cat([predicted_patches[:, :, i, :].transpose(1, 2) for i in range(output_patches)], dim=1)
        
        prediction = prediction[:, :self.config.output_len, :]
        if self.use_revin:
            prediction = self.revin(prediction, "denorm")
        
        result = {"prediction": prediction}
        if self.output_attentions:
            result["attn_weights"] = attn_weights
        if energy_loss is not None:
            result["loss"] = energy_loss
        return result
