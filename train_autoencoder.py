import torch
from src.basicts.models.PatchTST import PatchTSTAutoencoderForForecasting, PatchTSTConfig
from src.basicts.configs import BasicTSForecastingConfig
from src.basicts.runners.callback import EarlyStopping
from src.basicts import BasicTSLauncher


# Custom reconstruction metrics (compare inputs vs prediction, not targets vs prediction)
def reconstruction_mae(prediction=None, inputs=None, **kwargs):
    """Reconstruction MAE: compare prediction with inputs (not targets)."""
    if inputs is None or prediction is None:
        return torch.tensor(0.0)
    return torch.mean(torch.abs(prediction - inputs))


def reconstruction_mse(prediction=None, inputs=None, **kwargs):
    """Reconstruction MSE: compare prediction with inputs (not targets)."""
    if inputs is None or prediction is None:
        return torch.tensor(0.0)
    return torch.mean((prediction - inputs) ** 2)


def reconstruction_rmse(prediction=None, inputs=None, **kwargs):
    """Reconstruction RMSE: compare prediction with inputs (not targets)."""
    if inputs is None or prediction is None:
        return torch.tensor(0.0)
    return torch.sqrt(torch.mean((prediction - inputs) ** 2))


def run_experiment(dataset_name: str, num_features: int, input_len: int, output_len: int, patch_len: int = 4, 
        patch_stride: int = 4, latent_size: int = 16, num_encoder_layers: int = 2, num_decoder_layers: int = 2, 
        ae_dropout: float = 0.1, kl_weight: float = 1e-3, kl_clamp: float = 0.5, gpus: str = "0"):
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
    
    cfg = BasicTSForecastingConfig(
        model=PatchTSTAutoencoderForForecasting,
        model_config=model_config,
        dataset_name=dataset_name,
        input_len=input_len,
        output_len=output_len,
        gpus=gpus,
        batch_size=64,
        metrics=[
            ("ReconMAE", reconstruction_mae),
            ("ReconMSE", reconstruction_mse),
            ("ReconRMSE", reconstruction_rmse),
        ],
        target_metric="ReconMAE",  # Use reconstruction MAE for early stopping
        callbacks=[EarlyStopping(patience=10)],
        ckpt_save_dir=f"checkpoints/PatchTSTAutoencoder/{dataset_name}",
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
        "input_lens": [96],
        "output_lens": [96]
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
                    gpus=gpus,
                )