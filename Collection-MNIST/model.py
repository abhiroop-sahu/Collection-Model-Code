class CollectionNet(nn.Module):
  def __init__(self):
    super(CollectionNet, self).__init__()

    self.conv = nn.Conv2d(1, 10, 6)
    self.pl1 = nn.Linear(10*9*9, 50)
    self.pl2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(self.conv(x))
    x = torch.flatten(x, 1)
    x = F.relu(self.pl1(x))
    x = F.relu(self.pl2(x)) # the output
    return x

model = CollectionNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
