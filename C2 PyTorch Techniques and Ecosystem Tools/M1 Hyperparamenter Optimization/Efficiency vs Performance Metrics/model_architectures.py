import torch
import torch.nn as nn



class OptimizedCNN(nn.Module):
    """A Convolutional Neural Network with a specific, optimized architecture."""
    def __init__(self):
        """Initializes the layers of the CNN."""
        super(OptimizedCNN, self).__init__()

        # Define the hyperparameters for the model's architecture.
        n_layers = 2
        n_filters = [96, 128]
        kernel_sizes = [3, 5]
        dropout_rate = 0.10
        fc_size = 256

        # Define the feature extraction part of the network.
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, n_filters[0], kernel_sizes[0], padding=(kernel_sizes[0]-1)//2),
            nn.BatchNorm2d(n_filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Second convolutional block
            nn.Conv2d(n_filters[0], n_filters[1], kernel_sizes[1], padding=(kernel_sizes[1]-1)//2),
            nn.BatchNorm2d(n_filters[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Calculate the size of the feature map after it passes through the convolutional layers.
        feature_size = 32 // (2 ** n_layers)
        flattened_size = n_filters[-1] * feature_size * feature_size

        # Define the classification head of the network.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(flattened_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, 10)
        )

    def forward(self, x):
        """Defines the forward pass of the network.

        Args:
            x: The input tensor of shape (batch_size, channels, height, width).

        Returns:
            The output logits from the network.
        """
        # Pass the input through the feature extractor.
        x = self.features(x)
        # Flatten the feature map for the classifier.
        x = torch.flatten(x, 1)
        # Pass the flattened features through the classifier.
        x = self.classifier(x)
        # Return the final output logits.
        return x
    
    

class BasicBlock(nn.Module):
    """A basic residual block for ResNet, composed of two 3x3 convolutional layers."""
    # Class attribute defining the expansion factor for the number of output channels.
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """Initializes the layers of the BasicBlock.

        Args:
            in_planes: The number of input channels.
            planes: The number of output channels for the convolutional layers.
            stride: The stride for the first convolutional layer, used for downsampling.
        """
        super(BasicBlock, self).__init__()
        
        # Define the first convolutional layer followed by batch normalization.
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # Define the second convolutional layer followed by batch normalization.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Initialize an empty sequential module for the shortcut connection.
        self.shortcut = nn.Sequential()
        # If dimensions or stride change, a projection shortcut is needed to match dimensions.
        if stride != 1 or in_planes != self.expansion*planes:
            # The projection shortcut uses a 1x1 convolution to adjust channels and resolution.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        """Defines the forward pass through the BasicBlock.

        Args:
            x: The input tensor.

        Returns:
            The output tensor of the block.
        """
        # Pass input through the first convolution, batch norm, and ReLU activation.
        out = torch.relu(self.bn1(self.conv1(x)))
        # Pass through the second convolution and batch norm.
        out = self.bn2(self.conv2(out))
        # Add the output of the shortcut connection to the main path.
        out += self.shortcut(x)
        # Apply the final ReLU activation after the addition.
        out = torch.relu(out)
        # Return the output tensor.
        return out


class ResNet(nn.Module):
    """A generic ResNet model architecture."""
    def __init__(self, block, num_blocks, num_classes=10):
        """Initializes the ResNet model.

        Args:
            block: The type of block to use (e.g., BasicBlock).
            num_blocks: A list of four integers specifying the number of blocks
                        in each of the four layers.
            num_classes: The number of output classes for the final classifier.
        """
        super(ResNet, self).__init__()
        # Set the initial number of input planes for the first layer.
        self.in_planes = 64

        # Define the initial convolutional layer before the main residual blocks.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Create the four main stages of residual blocks.
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Define the final fully connected layer for classification.
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """Builds a ResNet layer composed of a specified number of residual blocks.

        Args:
            block: The block type to use in the layer.
            planes: The number of output channels for the blocks in this layer.
            num_blocks: The number of blocks to create in this layer.
            stride: The stride for the first block of the layer, used for downsampling.

        Returns:
            An nn.Sequential module representing the complete layer.
        """
        # The first block in a layer handles downsampling if its stride is not 1.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        # Iterate to create the required number of blocks.
        for stride in strides:
            # Append a new block to the list of layers.
            layers.append(block(self.in_planes, planes, stride))
            # Update the number of input planes for the next block to be created.
            self.in_planes = planes * block.expansion
        # Return the complete layer as a sequential module.
        return nn.Sequential(*layers)

    def forward(self, x):
        """Defines the forward pass through the ResNet model.

        Args:
            x: The input tensor of shape (batch_size, channels, height, width).

        Returns:
            The output logits from the classification layer.
        """
        # Pass the input through the initial convolutional layer.
        out = torch.relu(self.bn1(self.conv1(x)))
        # Pass the feature map through each of the four residual layers.
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Apply global average pooling to the final feature map.
        out = nn.functional.avg_pool2d(out, 4)
        # Flatten the output for the linear layer.
        out = out.view(out.size(0), -1)
        # Pass the features through the final classification layer.
        out = self.linear(out)
        # Return the output logits.
        return out
    
    
def ResNet34():
    """A factory function that constructs a ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3])