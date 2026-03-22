import numpy as np
from utils_RPTO import plot_car_ShapeNet_pressure
import os

size=np.load("/Users/zhangbojun/ML_Turbulent/RopeTransolver/data_ShapeNet_norm/pressure/x.npy")
x = np.load("/Users/zhangbojun/ML_Turbulent/RopeTransolver/data_ShapeNet_norm/pressure/x.npy")
y = np.load("/Users/zhangbojun/ML_Turbulent/RopeTransolver/data_ShapeNet_norm/pressure/y.npy")
y_hat = np.load("/Users/zhangbojun/ML_Turbulent/RopeTransolver/data_ShapeNet_norm/pressure/y_hat.npy")

size = size[0]
x = x[0]       # (10000,3)
y = y.squeeze(0)
y_hat = y_hat[0]    # (10000,)

print(x.shape)
print(y.shape)
print(y_hat.shape)

figure_path = "/Users/zhangbojun/ML_Turbulent/RopeTransolver/fig/pressure_ShapeNet_norm/"
plot_car_ShapeNet_pressure(
    size,
    x, y, y_hat,
    save_path=figure_path,
    figsize=(12, 8),
    colorbar_fontsize=24,
)
# true 20
# error 22