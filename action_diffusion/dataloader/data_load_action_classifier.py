import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class ActionDataset(Dataset):
    def __init__(self,root,):

        self.data_root = root
        self.file_name_list = os.listdir(self.data_root)
        self.file_path_list = [os.path.join(self.data_root, name) for name in self.file_name_list if name.endswith('.npy')]
        self.num_classes = 48
        self.data = []
        self._load_data()

    def _load_data(self):
        for d in self.file_path_list:
            path = os.path.join(self.data_root, d)
            data = np.load(path, allow_pickle=True)
            for info in data:
                action_labels = info['steps_ids']
                video_feature = info['video_feature']
                text_feature = info['frames_feature']
                self.data.append((action_labels, video_feature, text_feature))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        action_labels, video_feature, text_feature = self.data[idx]
        action_labels = torch.tensor(action_labels, dtype=torch.long)
        action_labels = F.one_hot(action_labels, num_classes=self.num_classes).float()
        video_feature = torch.tensor(video_feature, dtype=torch.float32)
        text_feature = torch.tensor(text_feature, dtype=torch.float32)

        return action_labels, video_feature, text_feature

