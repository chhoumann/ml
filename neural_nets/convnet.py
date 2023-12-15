import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms  # type: ignore


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        # 1 input channels for grayscale imgs, 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 32 in channels, 64 out channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # max pooling layer with 2x2 window
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # fully connected layer, 64*7*7 input feats, 128 out
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # fc layer w. 128 in feats, 10 out feats (10 classes)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # first conv layer with ReLU af
        x = F.relu(self.conv1(x))
        # max pooling
        x = self.pool(x)
        # second conv layer w. ReLU af
        x = F.relu(self.conv2(x))
        # max pooling
        x = self.pool(x)
        # flatten tensor for fc layer
        x = x.view(-1, 64 * 7 * 7)
        # apply first fc layer with ReLU af
        x = F.relu(self.fc1(x))
        # apply output fc layer
        x = self.fc2(x)
        return x


"""
Define a basic ConvNet with two convolutional layers followed by max pooling,
and then two fully connected layers.

Designed for input images of size 28x28 -- the size of MNIST dataset images.

Can be trained with a suitable optimizer, loss function, and dataset.

---
torch.nn provides classes and functions to build neural networks.
torch.nn.functional contains functions to be used in NNs, e.g. activation fns.

We use nn.Module as base class.
It comes from PyTorch and is the base class for all NNs.
We use __init__ to initialize the layers of the NN.

nn.Conv2d is a 2D convolutional layer.
The first input argument is for the input channels.
We use 1 for the first conv layer because MNIST images are grayscale.
For conv2 it's 32 because conv1 outputs 32 channels.
The second argument is the amount of output channels.
kernel_size is the size of the filter or kernel used in the convolution.
stride is the step size the kernel moves each time.
stride=1 means it moves 1 pixel at a time.
padding adds a border of 0s around the input.
Padding of 1 means 1px border is added,
    so spatial dimenions are equal after the convolution (since kernel_size=3)

MaxPool2d does max pooling with 2x2 window,
    reducing spatial dimensions (width & height) by half.

The fully connected layers:
- Layer 1 takes flattened output of last max-pooling layer.
    - Number of input features is calc'd by multiplying the number of output
    channels from the last conv layer (64) by the resulting image size (7x7)
- fc2 is the output layer, with 10 output features for the 10 classes in MINST

forward defines how the input `x` is passed through the network.
Each conv layer is followed by a relu and a max-pooling layer.
After conv layers, the data is flattened to pass through the fc layers.
Final output is the logits for each class,
    which can be passed through a softmax fn for classification.
"""

transform = transforms.Compose(
    [
        transforms.ToTensor(),  # convert imgs to tensors
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def train():
    model = SimpleConvNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass & optim
            optimizer.zero_grad()  # clear existing gradients
            loss.backward()  # compute gradients
            optimizer.step()  # update params

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Step [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

    return model


if __name__ == "__main__":
    model = train()
    model.eval()  # set to eval mode

    with torch.no_grad():  # inference withut gradient calculation
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Accuracy of the model on the 10000 test images: {100 * correct / total} %"
        )
