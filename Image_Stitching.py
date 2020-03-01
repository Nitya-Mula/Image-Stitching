import cv2
import numpy as np
import getopt
import sys
import random
# How to run:
# Open the environment in anaconda
# jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10

class MatchedObject:
    distance = 0
    img1_index = 999
    img2_index = 999

class Stitching:
    #  step 1
    def readImage(self,filename):
        img = cv2.imread(filename, 0)
        if img is None:
            print('No image found:' + filename)
            return None
        else:
            print('Image loaded...')

            return img

    #  step 2 and 3
    def findFeatures(self,img):
        descriptors = cv2.xfeatures2d.SIFT_create()
        (keypoints, descriptors) = descriptors.detectAndCompute(img, None)

        return keypoints, descriptors

    #  step 4 and 5
    def matchFeatures(self,kp1,kp2,desc1,desc2,img1,img2):
        threshold = 0.5
        norm_desc1 = []
        norm_desc2 = []
        matches = []
        for d1 in desc1:
            newd1 = d1 / 99
            norm_desc1.append(newd1)
        for d2 in desc2:
            newd2 = d2 / 99
            norm_desc2.append(newd2)
        for i in range(len(desc1)):
            for j in range(len(desc2)):
                distance = np.linalg.norm(norm_desc1[i]-norm_desc2[j],ord = 2)
                if(distance<threshold):
                    m = MatchedObject()
                    m.distance = distance
                    m.img1_index = i
                    m.img2_index = j
                    matches.append(m)
        print("\n Point pairs with distance less than threshold:")

        for x in matches:
            print("[({0}, {1}), ({2}, {3})]".format(int(kp1[x.img1_index].pt[0]),
                    int(kp1[x.img1_index].pt[1]),
                    int(kp2[x.img2_index].pt[0]),
                    int(kp2[x.img2_index].pt[1])))

        return matches

    def calculateHomography(self,correspondences):
        tempList = []
        for corr in correspondences:
            p1 = np.matrix([corr.item(0), corr.item(1), 1])
            p2 = np.matrix([corr.item(2), corr.item(3), 1])

            a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
            a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
            tempList.append(a1)
            tempList.append(a2)

        matrixA = np.matrix(tempList)

        u, s, v = np.linalg.svd(matrixA)

        h = np.reshape(v[8], (3, 3))

        h = (1/h.item(8)) * h

        return h

    def geometricDistance(self,correspondence, h):

        p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
        estimatep2 = np.dot(h, p1)
        estimatep2 = (1/estimatep2.item(2))*estimatep2

        p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
        error = p2 - estimatep2

        return np.linalg.norm(error)

    # step 6
    def ransac(self,corr, threshold):
        maxInliers = []
        finalH = None
        avgResiduefinal = 0

        for i in range(10):
            corr1 = corr[random.randrange(0, len(corr))]
            corr2 = corr[random.randrange(0, len(corr))]
            randomFour = np.vstack((corr1, corr2))
            corr3 = corr[random.randrange(0, len(corr))]
            randomFour = np.vstack((randomFour, corr3))
            corr4 = corr[random.randrange(0, len(corr))]
            randomFour = np.vstack((randomFour, corr4))

            h = self.calculateHomography(randomFour)
            inliers = []
            avgResidue = 0

            for i in range(len(corr)):
                d = self.geometricDistance(corr[i], h)
                avgResidue = avgResidue + d

                if d < 5:
                    inliers.append(corr[i])

            avgResidue = avgResidue / len(corr)

            if len(inliers) > len(maxInliers):
                maxInliers = inliers
                finalH = h
                avgResiduefinal = avgResidue

            if len(maxInliers) > (len(corr)*threshold):
                break
        
        print("\n Average residual of inliers for the best fit  {0} ".format(avgResiduefinal))
            
        return finalH, maxInliers
    

    def drawMatches(self,img1, kp1, img2, kp2, matches, inliers = None):
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]
        rows2 = img2.shape[0]
        cols2 = img2.shape[1]

        out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
        out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])
        out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])
        for mat in matches:

            img1_idx = mat.img1_index
            img2_idx = mat.img2_index

            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt

            inlier = False

            if inliers is not None:
                for i in inliers:
                    if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                        inlier = True

            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

            if inliers is not None and inlier:
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
            elif inliers is not None:
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

            if inliers is None:
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

        return out

    def main(self,img1name,img2name):
        estimation_thresh = 1

        img1 = self.readImage(img1name)
        img2 = self.readImage(img2name)

        correspondenceList = []
        if img1 is not None and img2 is not None:
            kp1, desc1 = self.findFeatures(img1)
            kp2, desc2 = self.findFeatures(img2)
            print ("Found keypoints in {0} : {1}" .format(img1name, str(len(kp1))))
            print ("Found keypoints in {0} : {1}" .format(img2name, str(len(kp2))))
            keypoints = [kp1,kp2]
            matches = self.matchFeatures(kp1, kp2, desc1, desc2, img1, img2)
            for match in matches:
                (x1, y1) = keypoints[0][match.img1_index].pt
                (x2, y2) = keypoints[1][match.img2_index].pt
                correspondenceList.append([x1, y1, x2, y2])

            corrs = np.matrix(correspondenceList)

            finalH, inliers = self.ransac(corrs, estimation_thresh)
            print ("Final homography: {0}".format(finalH))
            print ("Final inliers count: {0}".format(len(inliers)))

            matchImg = self.drawMatches(img1,kp1,img2,kp2,matches,inliers)

        return (finalH,matchImg)

    def getfinalImage(self, img1, img2, Homography):

        totalWidth = img1.shape[1] + img2.shape[1]
        finalImage = cv2.warpPerspective(img1, Homography, (totalWidth , img2.shape[0]))
        finalImage[0:img2.shape[0], 0:img2.shape[1]] = img2
        
        return finalImage