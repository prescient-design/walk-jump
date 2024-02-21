import hydra
import torch
import wandb
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from walkjump.cmdline.utils import instantiate_redesign_mask, instantiate_seeds, instantiate_model_for_sample_mode
from walkjump.sampling import walkjump


@hydra.main(version_base=None, config_path="../hydra_config", config_name="sample")
def sample(cfg: DictConfig) -> bool:
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    wandb.require("service")
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    hydra.utils.instantiate(cfg.setup)

    if cfg.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print(f"using device {cfg.device}")
        device = torch.device(cfg.device)

    mask_idxs = instantiate_redesign_mask(cfg.designs.redesign_regions or [])
    seeds = instantiate_seeds(cfg.designs)

    if not cfg.dryrun:
        model = instantiate_model_for_sample_mode(cfg.model).to(device)
        sample_df = walkjump(
            seeds,
            model,
            delta=cfg.langevin.delta,
            lipschitz=cfg.langevin.lipschitz,
            friction=cfg.langevin.friction,
            steps=cfg.langevin.steps,
            num_samples=cfg.designs.num_samples,
            mask_idxs=mask_idxs,
            chunksize=cfg.designs.chunksize,
        )
        sample_df.drop_duplicates(subset=["fv_heavy_aho", "fv_light_aho"], inplace=True)
        print(f"Writing {len(sample_df)} samples to {cfg.designs.output_csv}")
        sample_df.to_csv(cfg.designs.output_csv, index=False)
    
    return True
