import torch.nn as nn
import torch
import torchvision.models as models
import geffnet

# AlexNet
# VGG-11,13,16,19
# vGG-11,13,16,19 with batch normalization
# ResNet-18,34,50,101,152
# SquuezeNet 1.0/1.1
# Densenet-121,169,201,161
# Inception v3
# GoogleNet
# ShuffleNetV2
# MobileNet V2
# ResNext-50-32x4d, 101-32x8d
# Wide ResNet-50-2/ Wide ResNet-101-2
# MNASNet 1.0

def model_import(name):
    if name.startswith('ResNet'):
        if name == 'ResNet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, 100)
        elif name == 'ResNet34':
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(512, 100)
        elif name == 'ResNet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(2048, 100)
        elif name == 'ResNet101':
            model = models.resnet101(pretrained=True)
            model.fc = nn.Linear(2048, 100)
        elif name == 'ResNet152':
            model = models.resnet152(pretrained=True)
            model.fc = nn.Linear(2048, 100)
        elif name == 'Wide_ResNet50_2':
            model = models.wide_resnet50_2(pretrained=True)
            model.fc = nn.Linear(2048, 100)
        elif name == 'Wide_ResNet101_2':
            model = models.wide_resnet101_2(pretrained=True)
            model.fc = nn.Linear(2048, 100)
        
    elif name.startswith('VGG'):
        if name == 'VGG11':
            model = models.vgg16(pretrained = True)
        elif name == 'VGG13':
            model = models.vgg16(pretrained = True)
        elif name == 'VGG16':
            model = models.vgg16(pretrained = True)
        elif name == 'VGG19':
            model = models.vgg16(pretrained = True)
        elif name == 'VGG11_bn':
            model = models.vgg16_bn(pretrained = True)
        elif name == 'VGG13_bn':
            model = models.vgg16_bn(pretrained = True)
        elif name == 'VGG16_bn':
            model = models.vgg16_bn(pretrained = True)
        elif name == 'VGG19_bn':
            model = models.vgg16_bn(pretrained = True) 
        model.classifier[6] = nn.Linear(4096,100)
    elif name.startswith('MobileNet'):
        model = models.mobilenet_v2
        model.classifier[-1] = nn.linear(in_features=model.classifier[-1].in_features, out_features=100)
    elif name.startswith("EfficientNet"):
        if name=="EfficientNet_B3":
            model = geffnet.create_model('tf_efficientnet_b3', pretrained=True, num_classes=100)
        elif name=="EfficientNet_B2":
            model = geffnet.create_model('efficientnet_b2', pretrained=True, num_classes=100)
    return model
