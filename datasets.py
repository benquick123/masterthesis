import torch
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    
    def __init__(self, x, y, transforms=None):
        self.x = x
        self.y = y
        self.transforms = transforms
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        x_sample = self.x[idx]
        y_sample = self.y[idx]
        
        if self.transforms is not None:
            assert len(x_sample.shape) == 1, "Transforms assume only one sample is transformed at the time. There is also support just for grayscale images."
            x_sample = x_sample.view((int(x_sample.shape[0] ** (1/2)), -1))
            x_sample = self.transforms(x_sample)
            x_sample = x_sample.view(-1)
            
        return self.x[idx], self.y[idx]