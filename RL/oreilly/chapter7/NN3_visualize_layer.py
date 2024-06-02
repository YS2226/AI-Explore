import numpy as np
import dezero.layers as L
from dezero import Model
import dezero.functions as F

# Linearレイヤーの出力サイズ設定
linear = L.Linear(10)  # 出力サイズ設定

batch_size, input_size = 100, 5
x = np.random.randn(batch_size, input_size)
y = linear(x)

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # 入力の形状
        y = self.l1(x)
        print(f"After l1 (Linear layer) shape: {y.shape}")  # l1後の形状
        y = F.relu(y)
        print(f"After ReLU shape: {y.shape}")  # ReLU後の形状
        y = self.l2(y)
        print(f"After l2 (Linear layer) shape: {y.shape}")  # l2後の形状
        return y

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
lr = 0.2
iters = 10000

model = TwoLayerNet(10, 1)

for i in range(10):  # 10エポックまでに変更
    print(f"Iteration {i}")
    y_pred = model.forward(x)
    loss = F.mean_squared_error(y, y_pred)
    print(f"Loss: {loss.data}")
    model.cleargrads()
    loss.backward()
    for p in model.params():
        p.data -= lr * p.grad.data
    print()

# テストデータでの予測結果と元のデータをプロットして視覚化するためのコードを追加
import matplotlib.pyplot as plt

plt.scatter(x, y, color='blue', label='Data Points')
x_line = np.linspace(0, 1, 100).reshape(100, 1)
y_line = model.forward(x_line).data
plt.plot(x_line, y_line, color='red', label='Prediction Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

