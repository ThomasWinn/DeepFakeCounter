from torchvision import transforms
from torchvision.datasets import ImageFolder

class DataLoader():
    def __init__(self, file):
        self.stdev_file = file

    def _train_transform():
        return transforms.Compose(
            transforms.ToTensor()
        )

    def _valid_transform():
        return transforms.Compose(
            transforms.ToTensor()
        )

    def _test_transform():
        return transforms.Compose(
            transforms.ToTensor()
        )

    def _train_imagefolder(train_path):
        return ImageFolder(train_path)

    def _test_imagefolder(test_path):
        return ImageFolder(test_path)

    def get_train_dataloader():
        pass

    def get_valid_dataloader():
        pass

    def get_test_dataloader():
        pass