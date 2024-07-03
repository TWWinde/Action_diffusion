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
        self.index_map = self._create_index_map()

    def _create_index_map(self):
        index_map = []
        for file_idx, file_path in enumerate(self.file_path_list):
            data = np.load(file_path, allow_pickle=True)
            for sample_idx in range(len(data)):
                index_map.append((file_idx, sample_idx))
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        file_path = self.file_path_list[file_idx]
        data = np.load(file_path, allow_pickle=True)
        info = data[sample_idx]

        action_labels = info['steps_ids']
        print(action_labels)
        video_feature = info['video_feature']
        text_feature = info['text_feature']
        action_labels = torch.tensor(action_labels, dtype=torch.long)
        action_labels = F.one_hot(action_labels, num_classes=self.num_classes).float()
        video_feature = torch.tensor(video_feature, dtype=torch.float32)
        text_feature = torch.tensor(text_feature, dtype=torch.float32)

        return action_labels, video_feature, text_feature

