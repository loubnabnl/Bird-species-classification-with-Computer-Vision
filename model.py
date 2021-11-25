import torch.nn as nn
import torchvision.models as models


def define_resnet(num_layers, n_classes=20, freeze=True, use_cuda=False):
    """
    function that creates a ResNet model from a pretrained model on ImageNet
    we present two options:
        - training the whole network on the data and using the pretraining weights
        only as initialization
        - freezing the network and training only the last layer
    Arguments:
        -num_layers: integer giving number of layers in the ResNet
        -num_classes: integer, number of classes (in the output layer)
        -freeze: Boolean to indecate if we freeze the network or not
        -use_cuda: Boolean indicating whether to use GPU or not
    Return: ResNet model 
        """
        
    if num_layers == 50:
      model = models.resnet50(pretrained=True)
    elif num_layers == 101:
      model = models.resnet101(pretrained=True)
    elif num_layers == 152:
      model = models.resnet152(pretrained=True)
    else:
      ValueError('give valid model number')
    num_ftrs = model.fc.in_features
    if freeze:
      #freeze all except the last layer
      for param in model.parameters():
          param.requires_grad = False
      model.fc = nn.Linear(num_ftrs, n_classes)
    else:
      model.fc = nn.Linear(num_ftrs, n_classes)

    if use_cuda:
      print('using GPU')
      model.cuda()
    else:
        print('using CPU')
    return model
    