# Simplified Custom CNN Model
class ConvNet(nn.Module):
    def __init__(self, num_classes=102):
        super(ConvNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # (224x224x3) -> (224x224x32)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (224x224x32) -> (112x112x32)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (112x112x32) -> (112x112x64)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (112x112x64) -> (56x56x64)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (56x56x64) -> (56x56x128)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (56x56x128) -> (28x28x128)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (28x28x128) -> (28x28x256)
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (28x28x256) -> (14x14x256)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(14 * 14 * 256, 8192),  # Flatten layer (14x14x256 = 50176)
            nn.LeakyReLU(),
            nn.Linear(8192, 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_classes)  # Output layer (102 classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten before passing to FC layers
        x = self.fc_layers(x)
        return x
