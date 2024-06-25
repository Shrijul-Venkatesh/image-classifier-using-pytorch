from torch.utils.data import DataLoader


def loader(train_data, test_data):
    return {
        "train": DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
        "test": DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
    }
