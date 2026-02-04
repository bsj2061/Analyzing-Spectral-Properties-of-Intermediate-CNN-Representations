from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
