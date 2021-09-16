#imports
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class flowDataset(Dataset):
    """2 Phase FLow Dataset"""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.imgs_path = 'data/images/'
        self.csv_path  = r'data\data_labels.csv'
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)

        df = pd.read_csv(self.csv_path,index_col ="filename")

        self.data = []

        for class_path in file_list:
            for img_path in glob.glob(class_path+'/*.jpg'):
                file_name = img_path.split("\\")[-1]
                self.data.append([img_path, df.loc[file_name,'x']])
        #print(self.data)

        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        class_id = class_name

        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        
        return img_tensor, class_id

 
if __name__ == '__main__':
    dataset = flowDataset()
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2)
    for imgs, labels in dataloader:
        print("Batch of images has shape: ",imgs.shape)
        print("Batch of labels has shape: ", labels.shape)
