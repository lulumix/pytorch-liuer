import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

if __name__ == '__main__':
    dataset = DiabetesDataset("dataset/diabetes.csv.gz")
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

    model = Model()
    criterion = torch.nn.BCELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 保存损失值
    epoch_losses = []
    for epoch in range(100):
        epoch_loss = 0
        for i, data in enumerate(train_loader, 0):
            # 1.prepare data
            inputs, labels = data
            # 2.forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            epoch_loss += loss.item()
            # 3.backward
            optimizer.zero_grad()
            loss.backward()
            # 4.update
            optimizer.step()
        avg_loss = epoch_loss / len(train_loader)  # 计算每个epoch的平均损失
        epoch_losses.append(avg_loss)

    plt.plot(epoch_losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

