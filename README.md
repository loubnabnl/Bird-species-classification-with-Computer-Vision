## Birds Spieces Classification using Computer Vision

This project was part of the MVA kaggle challenge of the Object Recognition and Computer Vision course.
We tried to predict bird breeds using Computer Vision techniques.
The images were taken from the Caltech-UCSD Birds-200-2011 dataset.
Given 1187 images and 20 target categories, the goal is to build 
a model with the highest accracy on a test dataset containing the
same classes. <br>
To avoid overfitting and make use of the performant models that were 
trained on large datasets such as ImageNet, we used the concept of 
transfer learning \cite{btransfer}. We also did some preprocessing, 
before applying the model, using birds cropping and data augmentation
 techniques. This enabled us to detect bird breeds better and to enrich our dataset.

### Overview of the files
* Install packages using requirement.txt file.
* To perfrom cropping on the images using YOLOv3 for object detection model run crop_birds.py.
* main_file.py allows to train and save a ResNet101 model.
* evaluate.py allows you to evaluate your model on the test set and predicts classes of the images in this set.
* data.py contains code for data augmentation and images resizing.
* model.py has the code for creating ResNet models.
