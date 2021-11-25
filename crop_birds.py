import os
import json
import PIL.Image as Image
from chainercv.links import YOLOv3
from chainercv.utils import read_image

"""Using YOLO for bird detection and cropping as part of the data preprocessing"""

#we use pretrained YOLOv3 model on the voc0712 dataset 
model_crop = YOLOv3(pretrained_model='voc0712')

#model_crop.cuda()

def get_bird_boxes(img_path, model_crop, threshold = 0.85, for_test_set = False):
    """use of the model model_crop to get boxes around the birds in the image
    for_test_set is set as True when we do the cropping on the test set, we don't want to
    change the number of images"""
    
    img = read_image(img_path)
    bboxes, labels, scores = model_crop.predict([img])
    #if a box was detected for the image
    if len(labels)>0:
      #check if it's a bird box
      box_coord = []
      if for_test_set:
          #we want to replace the original image with one single image
          for i in range(len(labels[0])):
            #find the first bird that was detected
            #in the voc dataset on which the model was trained the bird label is at index 2
            if labels[0][i] == 2 and scores[0][i] > threshold:
              box_coord.append(bboxes[0][i])
              break
      else:
          for i in range(len(labels[0])):
            #in the voc dataset on which the model was trained the bird label is at index 2
            if labels[0][i] == 2 and scores[0][i] > threshold:
              box_coord.append(bboxes[0][i])
      return box_coord
    return []

def save_cropped_img(box_coord, img_path, count):
    """function to save the cropped images in the desired folder
    if birds are detected we remove original image and save new cropped 
    images of the birds, otherwise we leave the original image"""
    
    img = Image.open(img_path)
    ymin, xmin, ymax, xmax  = box_coord
    new_img = img.crop((xmin, ymin, xmax, ymax ))
    if count == 0:
      #replace original image
      new_img.save(img_path,optimize=True,quality=95)
    else:
      #we save other images in the same dir as the original image with a diffrent name
      new_path = img_path[:-4]+str(count)+img_path[-4:] 
      new_img.save(new_path,optimize=True,quality=95)


def crop_by_folders(path, model_crop, threshold=0.85, for_test_set = False):
    """function to crop bird images that exist in "path", 
    if a bird is detected we remove original images and save 
    the new cropped images in "path", for the test set we save at 
    maximum one cropped bird image for each image """
    
    #retrieve all images in the folder of path
    for root, dirs, files in os.walk(path):
      for i,name in enumerate(files):
        #crop all birds in the image in name
        img_path = os.path.join(root, name)
        print(img_path)
        if len(files)>30:
          if i%30 == 0:
            print('{} out of {} being processed'.format(i, len(files)))
        box_coord = get_bird_boxes(img_path, model_crop, threshold, for_test_set)
        if len(box_coord) == 0:
          print('no bird was detected')
        else:
          for ind in range(len(box_coord)):
            #save the cropped image in the same dir as the original image 
            save_cropped_img(box_coord[ind], img_path, ind)



""" Cropping of all images in the bird datasets"""
with open('config.json',) as file : 
    config = json.load(file)
    
paths = config['paths']
train_dir = paths['data'] + '/train_images'  
crop_by_folders(train_dir, model_crop, threshold=0.9)

val_dir = paths['data']  + '/val_images'
crop_by_folders(val_dir, model_crop, threshold=0.9)

test_dir = paths['data'] + '/test_images/mistery_category'
crop_by_folders(test_dir, model_crop, threshold=0.9)