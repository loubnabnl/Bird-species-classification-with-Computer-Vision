import json
import os
from tqdm import tqdm
import PIL.Image as Image
import torch

from model import define_resnet
from data import data_transforms

# settings
with open('config.json',) as file : 
    config = json.load(file)
    
use_cuda = torch.cuda.is_available()
#import the model file to be evaluated, it has to be in experiment folder
#change model name in config file to import the best model
state_dict = torch.load(config['paths']['model'])
model = define_resnet(101, 20, True, use_cuda)
model.load_state_dict(state_dict)
model.eval()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

test_dir = config['paths']['data'] + '/test_images/mistery_category'

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open('kaggle.csv', "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = data_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()


