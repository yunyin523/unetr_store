import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CMRDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', domain='', scale=0.30, rotate=180, transform=None):

        self.mode = mode
        self.dataset_dir = dataset_dir
        self.scale = scale
        self.rotate = rotate
        self.transform = transform

        if self.mode == 'train':
            pre_face = 'Training'

        elif self.mode == 'test':
            if 'A' in domain:
                pre_face = 'Testing/A' 
            elif 'B' in domain:
                pre_face = 'Testing/B'
            elif 'C' in domain:
                pre_face = 'Testing/C'
        elif self.mode == 'validation':
            pre_face = 'Validation'
        path = self.dataset_dir + pre_face + '/'
        print('start loading data')
        
        name_list = []

        df = pd.read_csv(path+"name_nii.csv")
        for index, row in df.iterrows():
            name = row["vendor"] + '_' + row["image_name"] + ".nii"
            name_list.append(name)
        rec_list = []
        for name in name_list:
            str_img = self.dataset_dir+"pre_processed/oct_imgs/"+name
            str_lab = self.dataset_dir+"pre_processed/oct_masks/"+name
            rec_list.append({'image': str_img, 'label':str_lab})
        self.name_list = name_list
        self.rec_list = rec_list


        print('load done, length of dataset:', len(rec_list))
        
    def __len__(self):
        return len(self.rec_list)


    def __getitem__(self, idx):
        ans = self.rec_list[idx]
        if self.transform:
            ans = self.transform(ans)
        return ans



