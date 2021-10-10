#imports
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision
from PIL import Image
from collections import Counter


class flowDataset(Dataset):
    """2 Phase FLow Dataset"""

    def __init__(self, val= False ,val_size = 0.2,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.validation_set = val
        self.val_size = val_size
        self.imgs_path = 'D:\Flow Videos\dataset'
        self.csv_path  = 'D:\Flow Videos\data_labels.csv'
        file_list = glob.glob(self.imgs_path + "*")
        #print(file_list)

        df = pd.read_csv(self.csv_path,index_col ="filename")

        self.datapath = []  # stores each image path
        self.label = [] # stores each label 

        self.transform = transform

        for class_path in file_list:
            for img_path in glob.glob(class_path+'/*.jpg'):
                file_name = img_path.split("\\")[-1]
                label = df.loc[file_name,'flow']
                self.label.append(label)
                self.datapath.append(img_path)  # flow is the column name in the csv
        #print(self.data)

        self.class_map = {'bubbly': 0,'slug':1,'churn':2,'annular':3}

        self.val = []
        self.train = []

        #Sort into test and train
        for key in self.class_map:
            
            indices = [i for i, x in enumerate(self.label) if x == key]
            split = int(np.floor(self.val_size * len(indices)))
            train_indices, val_indices = indices[split:], indices[:split]

            for i in train_indices:
                new = [self.datapath[i],self.label[i]]
                self.train.append( new)

            for i in val_indices:
                new = [self.datapath[i],self.label[i]]
                self.val.append( new)



        

    def __len__(self):

        if self.validation_set :
            return len(  self.val)
        else:
            return len( self.train)


    def __getitem__(self, idx):

        if self.validation_set :
            img_path, class_name = self.val[idx]
        else:
            img_path, class_name = self.train[idx]

        
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
