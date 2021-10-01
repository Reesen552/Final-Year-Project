#imports
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision
from PIL import Image


class flowDataset(Dataset):
    """2 Phase FLow Dataset"""

    def __init__(self,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.imgs_path = 'D:\Flow Videos\dataset'
        self.csv_path  = 'D:\Flow Videos\data_labels.csv'
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)

        df = pd.read_csv(self.csv_path,index_col ="filename")

        self.data = []
        self.transform = transform

        for class_path in file_list:
            for img_path in glob.glob(class_path+'/*.jpg'):
                file_name = img_path.split("\\")[-1]
                self.data.append([img_path, df.loc[file_name,'flow']])  # flow is the column name in the csv
        #print(self.data)

        self.class_map = {'bubbly': 0, 'bubbly-slug':1, 'slug':2, 'slug-churn':3, 'churn':4, 'churn-annular':5, 'annular':6}

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        class_id = self.class_map[class_name]

        img_tensor = img
        #img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor(class_id)
        
        return img_tensor.float(), class_id

 
if __name__ == '__main__':
    dataset = flowDataset()
    print("dataset size: ", len(dataset))
    img,label = dataset[10]

    print("Batch of images has shape: ",img.shape)
    print("Batch of labels has shape: ", label)

    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=0)
    for imgs, labels in dataloader:
          print("Batch of images has shape: ",imgs.shape)
          print("Batch of labels has shape: ", labels.shape)
