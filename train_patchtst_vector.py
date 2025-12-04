"""Train PatchTST_Vector model with pretrained autoencoder."""
import os
import glob
import torch
from typing import TYPE_CHECKING
from basicts.models.PatchTST import PatchTSTVectorForForecasting, PatchTSTConfig
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping, GradientClipping, BasicTSCallback
from basicts import BasicTSLauncher
from basicts.metrics import masked_mse

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


def energy_loss_fn(**kwargs) -> torch.Tensor:
    if "loss" in kwargs:
        return kwargs["loss"]
    prediction = kwargs.get("prediction") or kwargs.get("input")
    targets = kwargs.get("targets") or kwargs.get("target")
    if prediction is not None and targets is not None:
        return masked_mse(prediction, targets)
    raise ValueError("Loss function must receive either 'loss' from model or both 'prediction' and 'targets'")


class LoadAutoencoderWeights(BasicTSCallback):
    """Callback to load pretrained autoencoder weights."""
    
    def __init__(self, ae_ckpt: str):
        super().__init__()
        self.ae_ckpt = ae_ckpt
    
    def on_train_start(self, runner: "BasicTSRunner"):
        if not (self.ae_ckpt and os.path.exists(self.ae_ckpt)):
            return
        
        runner.logger.info(f"Loading pretrained autoencoder: {self.ae_ckpt}")
        checkpoint = torch.load(self.ae_ckpt, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        
        ae_state_dict = {}
        for key, value in state_dict.items():
            if "autoencoder" in key.lower():
                ae_state_dict[key.replace("model.", "").replace("backbone.", "")] = value
        
        if not ae_state_dict:
            return
        
        try:
            model = runner.model.module if hasattr(runner.model, "module") else runner.model
            if hasattr(model, "autoencoder"):
                model.autoencoder.load_state_dict(ae_state_dict, strict=False)
                runner.logger.info("Successfully loaded autoencoder weights")
            else:
                runner.logger.warning("Model does not have 'autoencoder' attribute")
        except Exception as e:
            runner.logger.warning(f"Failed to load autoencoder weights: {e}, continuing with random init")


def main():
    model_config = PatchTSTConfig(
        input_len=96, output_len=96, num_features=7,
        patch_len=16, patch_stride=8,
        hidden_size=256, latent_size=16,
        num_encoder_layers=2, num_decoder_layers=2,
        num_mlp_layers=4, num_samples=8, beta=1.0, noise_size=16,
        ae_dropout=0.1, kl_weight=1e-3, kl_clamp=0.5,
        num_layers=3, n_heads=4, intermediate_size=256 * 4, fc_dropout=0.1,
        use_revin=True, affine=True, subtract_last=False, decomp=False,
    )
    model_config.freeze_ae = True
    
    callbacks = [EarlyStopping(patience=10), GradientClipping(1.0)]
    ae_ckpt_dir = "checkpoints/PatchTSTAutoencoder"
    ckpt_files = glob.glob(os.path.join(ae_ckpt_dir, "**", "*_best_val_*.pt"), recursive=True)
    if ckpt_files:
        callbacks.append(LoadAutoencoderWeights(sorted(ckpt_files, key=os.path.getmtime)[-1]))
    
    cfg = BasicTSForecastingConfig(
        model=PatchTSTVectorForForecasting, model_config=model_config,
        dataset_name="ETTh1", input_len=96, output_len=96,
        gpus="0", num_epochs=100, batch_size=64,
        metrics=["MSE", "MAE"], loss=energy_loss_fn,
        optimizer_params={"lr": 1e-3}, callbacks=callbacks,
        ckpt_save_dir="checkpoints/PatchTSTVector", seed=42,
    )
    
    BasicTSLauncher.launch_training(cfg)


if __name__ == "__main__":
    main()
