import numpy as np
import cv2
from matplotlib import pyplot as plt

def getTracker():
    #global tracker, ar_verts, ar_edges

    im = cv2.imread('d:/WORK/2015/2015_09_08 - multikoptera/pyc/lena.jpg', 0)
    # print(tag.shape)
    rect = [0, 0, im.shape[0], im.shape[1]]

    a = max(im.shape)
    # im = np.uint8(np.absolute(cv2.Laplacian(tag, cv2.CV_64F)))
    im = cv2.copyMakeBorder(im, a, a, a, a, cv2.BORDER_CONSTANT, value=255)
    # tracker.add_target(im, rect)
    cv2.imshow('ahoj',im)
    print("TAG tracker added here")

    return im

# img1 = cv2.imread('tag1.bmp',0)          # queryImage
img1 = getTracker()
# img2 = cv2.imread('tag1inScene.png',0) # trainImage
img2 = cv2.imread('lenaRot.jpg',0) # trainImage


# Initiate SIFT detector
sift = cv2.ORB_create( nfeatures = 300 )

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)


hDiff = img2.shape[0] - img1.shape[0]
img1a = cv2.copyMakeBorder(img1,0,hDiff,0,0,cv2.BORDER_CONSTANT, value=0)
img3 = cv2.cvtColor(np.hstack([img1a,img2]), cv2.COLOR_GRAY2RGB)

cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,outImg=img3,**draw_params )

plt.imshow(img3,),plt.show()