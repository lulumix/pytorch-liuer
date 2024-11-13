import torch

x_data = [1, 2, 3]
y_data = [2, 4, 6]

w = torch.Tensor([1.0])
w.requires_grad = True

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print(f"predict (before training): 4 {forward(4).item()}")

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print(f"grad:{x},{y},{w.grad.item()}")
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_() # zero the gradient, important!

    print("progress:", epoch, l.item())

print(f"predict (after training): 4 {forward(4).item()}")
