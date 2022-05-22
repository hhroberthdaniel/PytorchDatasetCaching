import os
import random
import shutil
import tempfile

import torch
from torch.utils.data import Dataset


class CacheDataset(Dataset):
    r"""A class that can adds caching functionality to any existing Torch Dataset, by saving a number of
    augmentation to disk`.

    Args:
        dataset: any Torch Dataset

        augmentations_per_sample : sets the number of augmentations that are saved for each sample.
            For example, if your dataset has 1k samples and you wish to generate 20 augmentations for each, it will
            result in a total of 20k distinct samples.

        cache_dir: the directory where to save the data

        reset_cache_dir: whether to delete any pre-existing generated samples from the cache dir. If you do not modify
            the transformations, this can be set to False to save even more time, because it will not need to regenerate
            the samples

        """
    def __init__(self, dataset: Dataset, augmentations_per_sample: int, cache_dir: str, reset_cache_dir: bool = True):
        self.reset_cache_dir = reset_cache_dir
        self.dataset = dataset
        self.augs_per_sample = augmentations_per_sample
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)
        if self.reset_cache_dir:
            shutil.rmtree(self.cache_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item_dir = os.path.join(self.cache_dir, str(idx))
        os.makedirs(item_dir, exist_ok=True)

        if len(os.listdir(item_dir)) >= self.augs_per_sample:
            sel_file = random.choice(os.listdir(item_dir))
            data = torch.load(os.path.join(item_dir, sel_file))
            return data
        else:
            data = self.dataset[idx]
            _, item_name = tempfile.mkstemp(suffix=".pth", dir=item_dir)
            torch.save(data, item_name)
            return data


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    import time

    AUGMENTATIONS_PER_SAMPLE = 2

    transform = transforms.Compose(
        [transforms.ToTensor(), ]
        +
        # simulating a compute intensive augmentation scheme
        [transforms.RandomRotation(degrees=10)] * 50)

    batch_size = 4

    # create a dummy dataset for setting the baseline
    trainset = torchvision.datasets.FakeData(size=100, image_size=(3, 128, 128), transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    print("Iterating a normal dataset with transformations")
    for e in range(2):
        start_time = time.time()
        for (data, label) in trainloader:
            pass
        stop_time = time.time()

        print(f"Time for epoch {e} : {stop_time - start_time}")


    # create the cached dataset to showcase the time improvement that will take affect after the samples
    # have been generated
    cached_trainset = CacheDataset(trainset, augmentations_per_sample=AUGMENTATIONS_PER_SAMPLE, cache_dir="./tmp")
    trainloader = torch.utils.data.DataLoader(cached_trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    print(f"Starting iterating with cache dataset. After reaching the selected number \\"
          f"of augmentations per second - {AUGMENTATIONS_PER_SAMPLE}, epochs should be faster")

    for e in range(AUGMENTATIONS_PER_SAMPLE + 3):
        start_time = time.time()
        for (data, label) in trainloader:
            pass

        stop_time = time.time()
        print(f"Time for epoch {e} : {stop_time - start_time}")

        if e == AUGMENTATIONS_PER_SAMPLE - 1:
            print(" ====== Augmentations have been cached ======= ")
