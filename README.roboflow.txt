
Women/Men - v1 2022-08-08 3:22pm
==============================

This dataset was exported via roboflow.com on May 9, 2023 at 2:02 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 2656 images.
- are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Fit (black edges))

The following augmentation was applied to create 3 versions of each source image:
* Random rotation of between -20 and +20 degrees
* Random shear of between -13° to +13° horizontally and -13° to +13° vertically
* Random brigthness adjustment of between -20 and +20 percent
* Random Gaussian blur of between 0 and 2 pixels
* Salt and pepper noise was applied to 8 percent of pixels


