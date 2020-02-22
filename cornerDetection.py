import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np

from scipy import signal
from PIL import Image
from scipy import ndimage as ndi


# step 1
def exact_Ep_d(img, pixel, theta):
  x, y = pixel[0 : 2] 

  offsets = { 0: (1, 0), 45: (1, 1), 90: (0, 1), 135: (-1, 1), 180: (-1, 0) }
  u, v = offsets[theta]

  Epd = img.getpixel( (x + u, y + v)) - img.getpixel( (x, y) )

  return Epd ** 2

# step 2
def exact_Ep(img, pixel, thetas = (0, 45, 90, 135, 180) ):
  EpMin = 9999999
  x, y = pixel[0 : 2]

  for theta in thetas:
    temp = exact_Ep_d(img, pixel, theta)
    if(temp < EpMin):
        EpMin = temp

  return EpMin

# step 3
def MoravecCornerDetection(img, threshold, kernel = [[0,0,0], [0,1,0], [0,0,0]]):
    final_corners = []

    (width, height) = img.size
    imageCorners = np.copy(img)
    imageEnergy = np.copy(img)
    
    imgE = Image.fromarray(imageEnergy)
    
    for w in range(1, width - 1):
        for h in range(1, height - 1):
            imgE.putpixel((w, h), exact_Ep(img, (w, h)))

    imgE = signal.convolve2d(imgE, kernel, mode = 'same', boundary = 'fill')

    for i in range(1, imgE.shape[1] - 1):
        for j in range(1, imgE.shape[0] - 1):
            if(imgE[j, i] > threshold):
                final_corners.append((j, i))

    return final_corners

# step 4
def harris_energy(img, kernel):
    FilterXSobel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    FilterYSobel = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    Ix = signal.convolve2d(img, FilterXSobel , mode='same')
    Iy = signal.convolve2d(img, FilterYSobel , mode='same')

    Ixy = ndi.gaussian_filter(Iy*Ix, sigma = 1)
    Ixx = ndi.gaussian_filter(Ix**2, sigma = 1)
    Iyy = ndi.gaussian_filter(Iy**2, sigma = 1)

    return [[Ixx, Ixy], [Ixy, Iyy]]

# step 5
def HarrisCornerDetection(img, kernel = "gaussian", lamda = 1 , threshold = 0):
    finalMatrix = harris_energy(img, kernel)

    Ixx = finalMatrix[0][0]
    Ixy = finalMatrix[0][1]
    Iyy = finalMatrix[1][1]

    traceOfMatrix = Ixx + Iyy
    detOfMatrix = Ixx * Iyy - Ixy ** 2

    HCD_response = detOfMatrix - lamda * traceOfMatrix
    R_max = np.amax(HCD_response)
    HCD_response_normalized = np.copy(HCD_response).astype(float)

    for i in range(len(HCD_response)):
        for j in range(len(HCD_response[0])):
            HCD_response_normalized[i, j] = HCD_response[i, j] / R_max
            
    HCD_corners = []

    for rowIDX, response in enumerate(HCD_response_normalized):
       for colIDX, R in enumerate(response):
           if R > threshold:
               HCD_corners.append( (rowIDX, colIDX) )

    return (HCD_response_normalized,HCD_corners)

# Extra Credit - Gaussian filter (separable)
def harris_energy_separable(img, kernel):
    FilterX = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    FilterY = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])

    Ix = signal.convolve2d(img, FilterX , mode='same')
    Iy = signal.convolve2d(img, FilterY , mode='same')

    Ixy = ndi.gaussian_filter(Iy*Ix, sigma = 1)
    Ixx = ndi.gaussian_filter(Ix**2, sigma = 1)
    Iyy = ndi.gaussian_filter(Iy**2, sigma = 1)

    return [[Ixx, Ixy], [Ixy, Iyy]]