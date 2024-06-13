import os
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

class ImageDataset(Dataset):
    def __init__(self, directory, target_image_size=128, enable_preprocessing = True):
        self.directory = directory
        self.filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]
        self.target_image_size = target_image_size
        self.size_filter_stats = {'valid': 0, 'invalid': 0}  # Initialize statistics
        self.enable_preprocessing = enable_preprocessing
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        while True:
            img_path = self.filenames[idx]
            img = Image.open(img_path).convert('RGB')
            try:
                if self.enable_preprocessing:
                    img = preprocess(img, target_image_size=self.target_image_size)
                    img = preprocess_vqgan(img)
                self.size_filter_stats['valid'] += 1  # Increment valid count
                return img
            except ValueError:
                self.size_filter_stats['invalid'] += 1  # Increment invalid count
                idx = (idx + 1) % len(self.filenames)

    def get_size_filter_stats(self):
        return self.size_filter_stats

class ImageBinaryDataset(Dataset):
    def __init__(self, directory):
        """
        Initialize the dataset with a directory containing image files.
        :param directory: Path to the directory containing image files.
        """
        self.directory = directory
        # List files ending with .jpg or .png
        self.filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        """
        Return the total number of files in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Retrieve the image file's binary data at the index `idx`.
        :param idx: Index of the file to retrieve.
        :return: Binary data of the image.
        """
        img_path = self.filenames[idx]
        with open(img_path, 'rb') as f:
            img_data = f.read()
        return img_data

def preprocess(img, target_image_size=128):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = T.ToTensor()(img)  # Remove torch.unsqueeze
    return img



def preprocess_vqgan(x):
    '''
    Why?
    '''
    x = 2. * x - 1.
    return x
def delete_small_images(directory, target_image_size):
    """
    Delete images in the specified directory whose height or width is less than the target image size.

    Args:
        directory (str): Path to the directory containing the images.
        target_image_size (int): The minimum size for both height and width of the images.
    """
    deleted_count = 0

    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.png')):  # Add other extensions as needed
            image_path = os.path.join(directory, filename)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    if width < target_image_size or height < target_image_size:
                        os.remove(image_path)
                        deleted_count += 1
                        print(f"Deleted {filename} (size: {width}x{height})")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                