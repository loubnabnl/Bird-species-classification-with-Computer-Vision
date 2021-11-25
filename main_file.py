import json
import os
import torch
import torch.optim as optim
from torchvision import datasets
from matplotlib import pyplot as plt

from data import data_transforms, transforms_for_augumentation
from model import define_resnet

# settings
with open('config.json',) as file : 
    config = json.load(file)
    
lr, momentum, nb_epochs = config['lr'], config['momentum'], config['nb_epochs']
batch_size, log_interval, seed = config['batch_size'], config['log_interval'], config['seed']
paths = config['paths']
data = paths['data']
experiment = paths['experiment']

use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)

# Create experiment folder
if not os.path.isdir(experiment):
    os.makedirs(experiment)
    
# Data loading

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(data + '/train_images',
                         transform=transforms_for_augumentation),
                         batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(data + '/val_images',
                         transform=data_transforms),
                         batch_size=batch_size, shuffle=False)


def train(model, optimizer, epoch, train_loss, train_acc, use_cuda):
    """function for training a model and saving the loss and accuracy
    in lists train_loss and train_acc"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        #criterion = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
        if isinstance(output, tuple):
            loss = sum((criterion(o,target) for o in output))
        else:
            loss = criterion(output, target)
        #loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #index of max proba
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    train_loss.append(running_loss/len(train_loader.dataset))
    train_acc.append(100. * correct / len(train_loader.dataset))
    
def validation(model, optimizer, val_loss, val_acc, use_cuda):
    """function for evaluating a model on a validation set and saving the loss and accuracy"""
    
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    val_loss.append(validation_loss)
    val_acc.append(100. * correct / len(val_loader.dataset))
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc):
    """function to plot the traing and validation losses
    and the training and validation accuracies for monitoring
    all inputs are lists"""
    
    f, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.plot(train_loss, label='train')
    ax1.plot(val_loss, label='validation')
    ax1.set_title('Train - Validation Loss')
    ax1.set_xlabel('num_epochs')
    ax1.set_ylabel('loss')
    ax1.legend(loc='best')
    
    ax2.plot(train_acc, label='train')
    ax2.plot(val_acc, label='validation')
    ax2.set_title('Train - Validation Accuracy')
    ax2.set_xlabel('num_epochs')
    ax2.set_ylabel('accuracy')
    ax2.legend(loc='best')
    
# Application:
    
#import model
model = define_resnet(101, 20, True, use_cuda)

optimizer = optim.SGD(model.parameters(), lr=lr,momentum=momentum)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

train_loss, val_loss, train_acc, val_acc = [], [], [], []
for epoch in range(1, nb_epochs + 1):
    train(model, optimizer, epoch, train_loss, train_acc, use_cuda)
    validation(model, optimizer, val_loss, val_acc, use_cuda)
    #scheduler.step()
    model_file = 'experiment' + '/model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)

plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc)
