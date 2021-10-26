import torch
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

import tqdm.auto as tqdm

class BalancedSubsetBatchSampler(BatchSampler):

    def __init__(self, dataset, n_classes, n_samples, indices):
        self.indices = np.asarray(indices)
        loader = DataLoader(Subset(dataset, indices)) # Getting the subset deterministically.
        
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: self.indices[np.where(self.labels.numpy() == label)[0]]
                                 for label in self.labels_set}
        
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            
            yield indices
            
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        # return len(self.dataset) // self.batch_size
        return len(self.indices)
