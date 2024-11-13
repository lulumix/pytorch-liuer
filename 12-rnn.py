import torch
from torch import nn

batch_size = 1
seq_len = 5
input_size = 4
hidden_size = 4
num_layers = 1
'''
input_size = 4
hidden_size = 4
batch_size = 1
'''
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 3, 3]
y_data = [3, 1, 2, 3, 2]

ont_hot_lookup = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]
x_one_hot = [ont_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data)
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers,self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)

    # def init_hidden(self):
    #     return torch.zeros(self.batch_size, self.hidden_size)

net = Model(input_size, hidden_size, batch_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
'''
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    print("Predicted string: ", end='')
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    loss.backward()
    optimizer.step()
    print(', Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 15, loss.item()))
'''
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[i] for i in idx]), end='')
    print(', Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 15, loss.item()))
