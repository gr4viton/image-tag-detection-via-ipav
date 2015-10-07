import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as mcm
import matplotlib as mpl

# def drawMatches(img1, kp1, img2, kp2, matches):
#     """
#     My own implementation of cv2.drawMatches as OpenCV 2.4.9
#     does not have this function available but it's supported in
#     OpenCV 3.0.0
#
#     This function takes in two images with their associated
#     keypoints, as well as a list of DMatch data structure (matches)
#     that contains which keypoints matched in which images.
#
#     An image will be produced where a montage is shown with
#     the first image followed by the second image beside it.
#
#     Keypoints are delineated with circles, while lines are connected
#     between matching keypoints.
#
#     img1,img2 - Grayscale images
#     kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
#               detection algorithms
#     matches - A list of matches of corresponding keypoints through any
#               OpenCV keypoint matching algorithm
#     """
#
#     # Create a new output image that concatenates the two images together
#     # (a.k.a) a montage
#     rows1 = img1.shape[0]
#     cols1 = img1.shape[1]
#     rows2 = img2.shape[0]
#     cols2 = img2.shape[1]
#
#     out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
#
#     # Place the first image to the left
#     out[:rows1,:cols1] = np.dstack([img1, img1, img1])
#
#     # Place the next image to the right of it
#     out[:rows2,cols1:] = np.dstack([img2, img2, img2])
#
#     # For each pair of points we have between both images
#     # draw circles, then connect a line between them
#     for mat in matches:
#
#         # Get the matching keypoints for each of the images
#         img1_idx = mat.queryIdx
#         img2_idx = mat.trainIdx
#
#         # x - columns
#         # y - rows
#         (x1,y1) = kp1[img1_idx].pt
#         (x2,y2) = kp2[img2_idx].pt
#
#         # Draw a small circle at both co-ordinates
#         # radius 4
#         # colour blue
#         # thickness = 1
#         cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
#         cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
#
#         # Draw a line in between the two points
#         # thickness = 1
#         # colour blue
#         cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
#
#
#     # Show the image
#     cv2.imshow('Matched Features', out)
#     cv2.waitKey(0)
#     cv2.destroyWindow('Matched Features')
#
#     # Also return the image if you'd like a copy
#     return out

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

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

def gaussIt(im):
    a = 15
    return cv2.blur(im,(a,a))

if __name__ == '__main__':
    bgColor = 0 # black
    # bgColor = 255 # white

    img1 = makeBorder(readIm('invnoborder', '2L'), bgColor)
    img2 = readIm('space', '2L')

    img1 = rotate(img1,180)
    # img1 = rotate(img1,90)

    # img1 = makeLaplacian(img1)
    # img2 = makeLaplacian(img2)


    # img1 = gaussIt(img1)
    # img2 = gaussIt(img2)

    # Initiate ORB detector
    # orb = cv2.ORB_create( nfeatures = 300 )
    orb = cv2.ORB_create( nfeatures = 400 )

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch( des1,des2, k=2)

    # Apply ratio test
    good = []
    goodIdx = []
    idx = 0
    const = 0.80
    # const = 0.90

    for m,n in matches:
        if m.distance < const*n.distance:
            good.append([m])
            goodIdx.append(idx)
        idx += 1

    hDiff = img2.shape[0] - img1.shape[0]
    img1a = cv2.copyMakeBorder(img1,0,hDiff,0,0,cv2.BORDER_CONSTANT, value=0)

    imgBoth = cv2.cvtColor(np.hstack([img1a,img2]), cv2.COLOR_GRAY2RGB)
    # img3 = []

    # Draw first 10 matches.
    # cv2.drawMatchesKnn(img1,kp1,img2,kp2, matches[:10], outImg=img3, flags=2)
    cv2.drawMatchesKnn(img1,kp1,img2,kp2, good, outImg=imgBoth, flags=2)

    # img3 = drawMatches(img1,kp1,img2,kp2,matches[:20])



    print(str(len(des1)) + " = number of tag descriptor")
    print(str(len(des2)) + " = number of descriptor in scene image" )
    print(str(len(matches)) + "= all matches" )
    print(str(len(good)) + " = good matches")

    print goodIdx
    MIN_MATCH_COUNT = 4

    if len(good)>MIN_MATCH_COUNT:
        # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        src_pts = np.float32([ kp1[m].pt for m in goodIdx ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m].pt for m in goodIdx ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # M = np.array([[1,0,0],[0,1,0],[0,0,1]])
        dst = cv2.perspectiveTransform(pts, M)

        # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        # img3 = cv2.polylines(img2,[np.int32(dst)],True,128,3, cv2.LINE_AA)
        img3 = cv2.polylines(img2,[np.int32(dst)],True,128,3, cv2.LINE_8)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    # p0, p1 = np.float32((p0, p1))
    # H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)

    print(str(M) + " = Homeography transformation matrix")

    img3 =  cv2.cvtColor(img3 , cv2.COLOR_GRAY2RGB)
    imgAll = np.hstack([imgBoth,img3])
    plt.imshow(imgAll, cmap = mpl.cm.Greys_r)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.show()