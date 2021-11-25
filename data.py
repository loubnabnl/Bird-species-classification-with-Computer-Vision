import torchvision.transforms as transforms

# we resize the images to 320 x 320 to have a good resolution
# we normalize the images to mean = 0 and standard-deviation = 1

#to do data augmentation we apply random cropping, flipping and perspective transformations


transforms_for_augumentation = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

data_transforms = transforms.Compose([
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])