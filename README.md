# Image_Augmentation
scripts to augment labelded images with bounding boxes

The following Python packages are needed:
- numpy
- pandas 
- opencv
- imgaug
- matplotlib

How to use the code:
1. run the script scripts/aug_images.py
2. visualize random result by running scripts/show_augmented_image_box.py

What you have to do to run it with your own images:
- add a new directory to images with .jpg files
- add a .csv file to annotations with the same name as the new image folder
  - the csv file must follow the PASCAL VOC Format. 
- run the files like above. 
