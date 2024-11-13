

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1, 2, 3]
y_data = [2, 4, 6]
def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred -y) * (y_pred -y)

w_list = []
b_list = []
mse_list = []
cnt = 0
for w in np.arange(0.0 ,4.1, 0.2):
    for b in np.arange(-2.0, 2.1, 0.2):
        print(f"w = {w} b = {b}")
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print("\t", x_val, y_val, y_pred_val, loss_val)
        print("MSE=", l_sum / len(x_data))
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / len(x_data))
        cnt += 1
print(f"一共多少次 = {cnt}")

# plt.plot(w_list, mse_list)
# plt.ylabel('loss')
# plt.xlabel('w')
# plt.show()

x = np.array(w_list)
y = np.array(b_list)
z = np.array(mse_list)


# 3D线图
fig = plt.figure()
ax =fig.add_axes(Axes3D(fig))
ax.plot(x, y, z)
plt.show()
