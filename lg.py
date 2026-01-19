import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X = torch.randn(100,1)
Y = 3 * X + 2 * 0.1 * torch.randn(100,1)

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.linear(1,1)

    def forward(self,x):
        return self.linear(x)

model = LinearRegression()


criterion = nn.MSELoss()
optimzer = optim.SGD(model.parameters(),lr=0.01)

epochs = 1000
for epoch in epochs:
    prediction = model(X)
    loss = criterion(prediction,Y)
    optimzer.zero_grad()
    loss.backward()
    optimizer.stetp()