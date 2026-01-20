import torch
import torch.nn as nn
import torch.optim as optim

X = torch.randn(100,3)

gradients = torch.tensor([[2.0],[-3.0],[1.3]])
bias = 4
Y = X *(gradients) + bias + 0.1 * torch.randn(100,1)

class MultipleRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
    def forward(self,x):
        self.linear(x)

criterion = nn.MSELoss()
model = MultipleRegression()
optimizer = optim.SGD(model.parameters(),lr= 0.01)

epochs = 1000
for i in range(0,epochs):
    predictions = model(X)
    loss = criterion(predictions, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if ((i+1)% epochs) == 0:
        print(f"Epochs is [{(i+1)/epochs}], Loss is {loss.item():.4f}")


weights = model.linear.weight.data
learned_bias = model.linear.bias.data
