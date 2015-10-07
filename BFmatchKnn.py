import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as mcm
import matplotlib as mpl

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

def readIm(pre, tag):
    dIm = 'd:/WORK/2015/2015_09_08 - multikoptera/pyc/pic/'
    fIm =  pre + '_' + tag + '.png'
    im = cv2.imread(dIm + fIm,0)
    if im is not None:
        print("Loaded image: [" + fIm + "] = " + str(im.shape) )
    return im

def makeBorder(im, bgColor):
    # rect = [0, 0, im.shape[0], im.shape[1]]
    a = max(im.shape)
    im = cv2.copyMakeBorder(im, a, a, a, a, cv2.BORDER_CONSTANT, value=bgColor)
    return im

def makeLaplacian(im):
    return np.uint8(np.absolute(cv2.Laplacian(im, cv2.CV_64F)))

bgColor = 0 # black
# bgColor = 255 # white

img1 = makeBorder(readIm('invnoborder', '2L'), bgColor)
img2 = readIm('space', '2L')


# img1 = makeLaplacian(img1)
# img2 = makeLaplacian(img2)

# tracker.add_target(im, rect)
#     print("TAG tracker added here")


# tag = getTracker()
# img1 = tag

# Initiate SIFT detector
# orb = cv2.ORB()
orb = cv2.ORB_create( nfeatures = 300 )

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
# matches = bf.match(des1,des2)
bf = cv2.BFMatcher()
print("Number of tag descriptor:")
print(len(des1))
print("Number of descriptor in scene image:")
print(len(des2))
matches = bf.knnMatch( des1,des2, k=2)

# Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])



hDiff = img2.shape[0] - img1.shape[0]
img1a = cv2.copyMakeBorder(img1,0,hDiff,0,0,cv2.BORDER_CONSTANT, value=0)

img3 = cv2.cvtColor(np.hstack([img1a,img2]), cv2.COLOR_GRAY2RGB)
# img3 = []

# Draw first 10 matches.
# cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img3, flags=1)
cv2.drawMatchesKnn(img1,kp1,img2,kp2, good, outImg=img3, flags=2)

# img3 = drawMatches(img1,kp1,img2,kp2,matches[:20])

plt.imshow(img3 , cmap = mpl.cm.Greys_r), plt.show()
