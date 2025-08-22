import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import  models
import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description="A script for dog cat classification.")
parser.add_argument("-i", "--image", type=str, help="Path to the input image")
args = parser.parse_args()
VGG_WEIGHTS_PATH = "./vgg19_trained.pth"


def get_vgg19_model(num_classes=2, freeze_features=True):
    # Load pre-trained VGG19
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    
    # Freeze feature extraction layers
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    
    model.classifier[6] = nn.Linear(4096, num_classes) 
    
    return model

vgg_model = get_vgg19_model(num_classes=2)

# Load weights
vgg_model.load_state_dict(torch.load(VGG_WEIGHTS_PATH, map_location=torch.device('cpu')))

# Predicting

vgg_model.eval()

# Take image from param


transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if args.image:
    image_path = args.image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms(image)
    image = image.unsqueeze(0)  # Add batch dimension
    prediction = vgg_model(image)  
    predicted_class = prediction.argmax(dim=1).item()
    if (predicted_class == 0):
        print("cat")
        class_label = "Cat"
    else:
        print("dog")
        class_label = "Dog"
    # Display the image with the predicted class
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))  

    plt.imshow(image_rgb)
    plt.axis('off') 

    plt.text(10, 30, f"Predicted: {class_label}", 
            fontsize=20, 
            color='red',  
            fontweight='bold',  
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))  
    plt.show()