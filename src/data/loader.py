from torch.utils.data import DataLoader

def build_loader(dataset, batch_size, shuffle, num_workers = 4, pin_memory = True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )