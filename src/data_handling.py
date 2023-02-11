import os
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from typing import Tuple


class FileHandler:
    """
    base class for handling files
    """
    def __init__(self, verbose: int = 0):
        """
        :param verbose: determines how much information is printed in the console
        :type verbose: int
        """
        self.verbose = verbose

    def names_to_ids(self, directory: str):
        """
        renames files in a directory to "0, 1, ...., N"

        :param directory: path to a directory containing files that have to be renamed
        :type directory: str
        """
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        for i, file in enumerate(files):
            old_file = os.path.join(directory, file)
            new_file = os.path.join(directory, str(i) + '.jpg')
            os.rename(old_file, new_file)
        if self.verbose > 0:
            print("replaced the names of", len(files), "files with int identifiers")


class ImageHandler(FileHandler):
    """
    base class for handling image data
    """
    def __init__(self, verbose: int = 0):
        """
        :param verbose: determines how much information is printed in the console
        :type verbose: int
        """
        super(ImageHandler, self).__init__(verbose=verbose)

    def make_vertical(self, directory: str):
        """
        makes image files within a directory vertical

        :param directory: path to a directory containing images that have to be made vertical
        :type directory: str
        """
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        counter_rotation = 0
        for file in files:
            file_path = directory + "/" + file
            image = Image.open(file_path)

            if image.width > image.height:
                image = image.rotate(90, expand=True)
                counter_rotation += 1
            image.save(file_path)

        if self.verbose > 0:
            print("rotated", counter_rotation, "images of", len(files), "to make them vertical")


class LOFFHandler(ImageHandler):
    """
    class for handling "litter on forest floor" data set
    """
    def __init__(self, root_directory: str = "../data/litter_on_forest_floor", verbose: int = 0):
        """
        :param root_directory: root directory of the "litter on forest floor" data set
        :type root_directory: str

        :param verbose: determines how much information is printed in the console
        :type verbose: int
        """
        super(LOFFHandler, self).__init__(verbose=verbose)
        self.root_directory = root_directory

    def create_loaders(self, batch_size: int = 8, image_size: int = 512) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        creates train, val and test data loader

        :param batch_size: size of the batches that the loaders output
        :type batch_size: int

        :param image_size: size to which the images are resized
        :type image_size: int

        :return: train loader, val loader, test loader
        :rtype: Tuple[DataLoader, DataLoader, DataLoader]
        """
        transform = Compose([Resize(size=image_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
        data = ImageFolder(root=self.root_directory, transform=transform)
        lens = [int(len(data)*0.8) + 1, int(len(data)*0.1), int(len(data)*0.1)]
        data_train, data_val, data_test = random_split(dataset=data, lengths=lens)
        loader_train = DataLoader(dataset=data_train, batch_size=batch_size)
        loader_val = DataLoader(dataset=data_val, batch_size=batch_size)
        loader_test = DataLoader(dataset=data_test, batch_size=batch_size)
        return loader_train, loader_val, loader_test

    def handle_data(self):
        """
        replaced the image neames by indices and makes all images vertical
        """
        dirs = ["clean", "litter"]
        for d in dirs:
            self.names_to_ids(directory=self.root_directory + "/" + d)
            self.make_vertical(directory=self.root_directory + "/" + d)
