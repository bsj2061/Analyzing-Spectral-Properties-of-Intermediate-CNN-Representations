import yaml
from src.data import *
from src.utils import *
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../../configs", config_name="defaults")
def main(cfg: DictConfig):

    device = cfg.device

    seed = cfg.seed
    set_seed(seed)

    data_cfg = cfg.dataset
    model_cfg = cfg.model
    exp_cfg = cfg.experiment
    loader_cfg = cfg.dataloader
        
    ds = imagenet100.ImageNet100(
        root=data_cfg["root"],
        split="train",
        train_dirs=data_cfg["shards"]["TRAIN_DIRS"],
        val_dirs=data_cfg["shards"]["VAL_DIRS"],
        transform=transforms.get_transforms(dataset_name="imagenet100")
    )
    
    loader = build_loader(ds, 
                          batch_size=loader_cfg["batch_size"],
                          shuffle=loader_cfg["shuffle"],
                          num_workers=loader_cfg["num_workers"],
                          pin_memory=loader_cfg["pin_memory"]
                          )
    
if __name__=="__main__":
    main()    