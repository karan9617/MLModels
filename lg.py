import torch
import torch.nn as nn
import torch.optim as optim
import numpy

torch.manual_seed(42)

X = torch.randn(100,1)
Y = 3 * X + 2 * 0.1 * torch.randn(100,1)

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)

model = LinearRegression()


criterion = nn.MSELoss()
optimzer = optim.SGD(model.parameters(),lr=0.01)

epochs = 1000
for epoch in range(0,epochs):
    prediction = model(X)
    loss = criterion(prediction,Y)
    optimzer.zero_grad()
    loss.backward()
    optimzer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

weight = model.linear.weight.item()
bias = model.linear.bias.item()

print("\nLearned parameters:")
print(f"Weight: {weight:.3f}")
print(f"Bias: {bias:.3f}")