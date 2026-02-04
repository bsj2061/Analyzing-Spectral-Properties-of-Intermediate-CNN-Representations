import yaml
from src.data import *
from src.utils import *

def main():
    
    
    with open('./configs/experiments/layerwise_spectral.yaml') as f:
        exp_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        
    with open('./configs/data/imagenet100.yaml') as f:
        data_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    ds = imagenet100.ImageNet100(
        root=data_cfg["data"]["root"],
        split="train",
        train_dirs=data_cfg["data"]["shards"]["TRAIN_DIRS"],
        val_dirs=data_cfg["data"]["shards"]["VAL_DIRS"],
        transform=transforms.get_transforms(dataset_name="imagenet100")
    )
    
    
if __name__=="__main__":
    main()    