CONFIG_PATH = "../src/walkjump/hydra_config"
TRAINER_OVERRIDES = [
    "++trainer.accelerator=cpu",
    "++data.csv_data_path=data/poas.csv.gz",
    "++dryrun=true",
]

SAMPLER_OVERRIDES = [
    "++designs.seeds=denovo",
    "++dryrun=true",
    "++designs.redesign_regions=[L1,L2,H1,H2]",
    "++model.checkpoint_path=last.ckpt",
]
