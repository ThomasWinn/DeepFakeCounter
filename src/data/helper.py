import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import tqdm

def compute_mean_and_std():
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)

        return d["mean"], d["std"]
    
    ## ASSUMING WE GET STD AND MEAN OF TRAIN DATASET
    folder = '../../dataset'
    # folder = 'dataset'
    folder += '/train'
    print(os.getcwd())
    ds = ImageFolder(
        folder, transform=transforms.Compose([transforms.ToTensor()])
    )
    print(ds)
    print(len)
    dl = DataLoader(
        ds, batch_size=1, num_workers=0
    )

    mean = 0.0
    # for i in dl:
    #     continue
    # wtf error
    for images in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)

    var = 0.0
    npix = 0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()

    std = torch.sqrt(var / (npix / 3))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std

if __name__ == "__main__":
    compute_mean_and_std()