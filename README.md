# Image-Stitching

The above program does Image Stitching of two images.

Flow: 
  - Extracting features from the images
  - Computing the descriptors of the images
  - Finding the pair of descriptors whose distance is less than given threshold
  - Computing homography matrix with the pairs of features extracted from above steps
  - Run RANSAC algorithm to fit a model
  - Warp the two images
  
 Steps to run:
  - python Image_Stitching.py
  - Run the ipython notebook to test with the input images
