import torch
from torchvision import models

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    #Load pretrained model
    arch = checkpoint['architecture']
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    
    #Load classifier and hyperparameters
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    epochs = checkpoint['epochs']
    optimizer_state = checkpoint['optimizer_state']
    
    return model, epochs, optimizer_state