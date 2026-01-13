import torchvision.models as models
import torch
from torchvision import models
from torchsummary import summary

# Load a pretrained model, for example, ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Print the model architecture
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)

# Print the summary, you need to provide the input size (channels, height, width)
summary(model, (3, 224, 224))
