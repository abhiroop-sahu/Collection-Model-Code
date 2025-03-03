class CollectionNet(nn.Module):
  def __init__(self):
    super(CollectionNet, self).__init__()

    self.conv1 = nn.Conv2d(3, 16, 4) # outputs 61x61x16
    self.conv2 = nn.Conv2d(16, 32, 4) # outputs 58x58x32
    self.pool1 = nn.MaxPool2d(2, 2) # outputs 29x29x32
    self.conv3 = nn.Conv2d(32, 64, 4) # outputs 26x26x64
    self.pool2 = nn.MaxPool2d(2, 2) # outputs 13x13x64
    self.pl1 = nn.Linear(13 * 13 * 64, 800)
    self.pl2 = nn.Linear(800, 224)
    self.pl3 = nn.Linear(224, 102)

  def forward(self, x):
    x = F.leaky_relu(self.conv1(x))
    x = F.leaky_relu(self.conv2(x))
    x = self.pool1(x)
    x = F.leaky_relu(self.conv3(x))
    x = self.pool2(x)
    x = torch.flatten(x, 1)
    x = F.leaky_relu(self.pl1(x))
    x = F.leaky_relu(self.pl2(x))
    x = F.leaky_relu(self.pl3(x)) # the output
    return x
