import os
import glob
import torch
from src.basicts.models.PatchTST import PatchTSTVectorForForecasting, PatchTSTConfig
from src.basicts.configs import BasicTSForecastingConfig
from src.basicts.runners.callback import EarlyStopping, GradientClipping, BasicTSCallback
from src.basicts import BasicTSLauncher


def loss(**kwargs):
    """Loss function that uses model's loss directly."""
    if "loss" in kwargs:
        return kwargs["loss"]
    raise ValueError("Model must return 'loss' in forward return")


class LoadAutoencoderWeights(BasicTSCallback):
    """Callback to load pretrained autoencoder weights."""
    
    def __init__(self, ae_ckpt: str):
        super().__init__()
        self.ae_ckpt = ae_ckpt
    
    def on_train_start(self, runner):
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


def run_experiment(dataset_name: str, num_features: int, input_len: int, output_len: int,
                   patch_len: int = 4, patch_stride: int = 4, latent_size: int = 16,
                   num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                   ae_dropout: float = 0.1, kl_weight: float = 1e-3, kl_clamp: float = 0.5,
                   freeze_ae: bool = True, ae_ckpt_dir: str = None, gpus: str = "0"):
    model_config = PatchTSTConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=num_features,
        patch_len=patch_len,
        patch_stride=patch_stride,
        latent_size=latent_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        ae_dropout=ae_dropout,
        kl_weight=kl_weight,
        kl_clamp=kl_clamp,
    )
    model_config.freeze_ae = freeze_ae
    
    callbacks = [EarlyStopping(patience=10), GradientClipping(1.0)]
    if ae_ckpt_dir:
        ckpt_files = glob.glob(os.path.join(ae_ckpt_dir, "**", "*_best_val_*.pt"), recursive=True)
        if ckpt_files:
            callbacks.append(LoadAutoencoderWeights(sorted(ckpt_files, key=os.path.getmtime)[-1]))
    
    cfg = BasicTSForecastingConfig(
        model=PatchTSTVectorForForecasting,
        model_config=model_config,
        dataset_name=dataset_name,
        input_len=input_len,
        output_len=output_len,
        gpus=gpus,
        batch_size=64,
        metrics=["MSE", "MAE"],
        loss=loss,
        callbacks=callbacks,
        ckpt_save_dir=f"checkpoints/PatchTSTVector/{dataset_name}",
        seed=42,
    )
    
    BasicTSLauncher.launch_training(cfg)


datasets = [
    ("ETTh1", 7),
    ("ETTh2", 7),
    ("ETTm1", 7),
    ("ETTm2", 7),
    ("Electricity", 321),
    ("Weather", 21),
    ("ExchangeRate", 8),
]

dataset_configs = {
    "default": {
        "input_lens": [96, 192, 336, 512],
        "output_lens": [96, 192, 336, 720]
    }
}

gpus = "0"
patch_len = 4
patch_stride = 4
latent_size = 16
num_encoder_layers = 2
num_decoder_layers = 2
ae_dropout = 0.1
kl_weight = 1e-3
kl_clamp = 0.5
freeze_ae = True
ae_ckpt_dir = "checkpoints/PatchTSTAutoencoder"

if __name__ == "__main__":
    for dataset_name, num_features in datasets:
        config = dataset_configs.get(dataset_name, dataset_configs["default"])
        input_lens = config["input_lens"]
        output_lens = config["output_lens"]
        
        for input_len in input_lens:
            for output_len in output_lens:
                run_experiment(
                    dataset_name=dataset_name,
                    num_features=num_features,
                    input_len=input_len,
                    output_len=output_len,
                    patch_len=patch_len,
                    patch_stride=patch_stride,
                    latent_size=latent_size,
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    ae_dropout=ae_dropout,
                    kl_weight=kl_weight,
                    kl_clamp=kl_clamp,
                    freeze_ae=freeze_ae,
                    ae_ckpt_dir=ae_ckpt_dir + f"/{dataset_name}",
                    gpus=gpus,
                )
