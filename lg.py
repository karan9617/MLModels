import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

X= torch.randn(100,1)
Y = 3 * X + 2 + 0.1 * torch.randn(100, 1)  # True relationship with noise


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)


model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)

epochs = 3000
for epoch in range(1,epochs):
    predictions = model(X)
    loss = criterion(predictions,Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch +1) %100 == 0:
        print(f"Epoch [{(epoch+1)/epochs}], Loss:{loss}")

weight = model.linear.weight.item()
bias = model.linear.bias.item()

print("\nLearned parameters:")
print(f"Weight: {weight:.3f}")
print(f"Bias: {bias:.3f}")