import os
from typing import Any
from random import randint, choice
import matplotlib.pyplot as plt, os, pandas as pd
import torch, torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# output chinese word
import matplotlib
matplotlib.rc("font",family='SimHei')

from aitrain.dataset.hand_writting_data import HWData
from aitrain.utils import DataPathEnum

# image = 64 * 64
class HWDataset(HWData, Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> Any:
        label, image_path = self.get_image_path_vs_lable(index)
        image_tensor = self.transform(Image.open(image_path))
        # target tensor
        target = torch.zeros((15))
        target[label] = 1.0

        return image_tensor, target, self.character_str[label]
    
    def get_random_item(self):
        index = randint(0, self.__len__() -1)
        return self.__getitem__(index), index


class TorchClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(64*64, 1000),
            nn.LeakyReLU(0.02),
            nn.Linear(1000, 100),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(100),
            nn.Linear(100, 15),
            nn.Softmax(dim = 1)
        )

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss = nn.BCELoss()
        
        self.dataset = HWDataset()
        self.counter = 0
        self.progress = []
        
        self.checkpoint_file = os.path.join(str(DataPathEnum.MODEL_CHECKPOINT_DIR), 
                                            "zh_hw_torch.pth")

    def forward(self, x):
        x = x.view(x.size(0), -1) # same as nn.flatten(x)
        return self.model(x)
    

    def _train(self, x, y):
        outputs = self.forward(x)
        loss = self.loss(outputs, y)

        # optimize
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.counter += 1
        if(self.counter % 10 == 0):
            self.progress.append(loss.item())

    def train(self, epochs: int):
        print('start train model...')
        for epochs in range(epochs):
            data_loader = DataLoader(self.dataset, batch_size=100, shuffle=True)
            for index, (data, target, target_char) in enumerate(data_loader):
                self._train(data, target)
        self._plot_progress()


    def random_eval_model(self):
        (data, target, _), index = self.dataset.get_random_item()
        
        self.dataset.plot_image(index)
        with torch.no_grad():
            output = self.forward(data)
        
        df = pd.DataFrame(output.detach().numpy()).T
        df.plot.barh(rot=0, legend=False, ylim=(0, 15), xlim=(0,1))
        
    def _plot_progress(self):
        df = pd.DataFrame(self.progress, columns = ["loss"])
        df.plot(title="counter:" + str(self.counter), ylim=(0, 1.0), figsize=(16,8), 
                alpha=0.5, marker=".", grid=True, yticks=(0,0.25,0.5))


    def save_model_state(self):
        torch.save(self.model.state_dict(), self.checkpoint_file)


    def load_model_state(self):
        self.model.load_state_dict(torch.load(self.checkpoint_file))        


    
    




