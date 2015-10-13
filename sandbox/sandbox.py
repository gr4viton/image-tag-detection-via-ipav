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


def orbDesc(im1,im2, imBoth):

    # Initiate ORB detector
    # orb = cv2.ORB_create( nfeatures = 300 )
    orb = cv2.ORB_create( nfeatures = 400 )

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(im1,None)
    kp2, des2 = orb.detectAndCompute(im2,None)

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

    # img3 = []

    # change kp points - useless i will not use knnMatch and orb..
    # for m in goodIdx:
    #     # kp1[m].pt = tuple(x+100 for x in kp1[m].pt)
    #     print(m)
    #     print(kp2[m].pt)
    #     kp2[m].pt = tuple(x*0.5+100 for x in kp2[m].pt)
    #     # kp2[m].pt = tuple([10*m, 10])
    #     print(kp2[m].pt)
        # kp1[m].pt += 10.0


    # Draw first 10 matches.
    # cv2.drawMatchesKnn(img1,kp1,img2,kp2, matches[:10], outImg=img3, flags=2)
    cv2.drawMatchesKnn(im1,kp1,im2,kp2, good, outImg=imBoth, flags=2)


    # img3 = drawMatches(img1,kp1,img2,kp2,matches[:20])

    # vytvor si dvojice sam - ze znalosti rohu ctverce

    print(str(len(des1)) + " = number of tag descriptor")
    print(str(len(des2)) + " = number of descriptor in scene image" )
    print(str(len(matches)) + "= all matches" )
    print(str(len(good)) + " = good matches")

    # print(goodIdx)
    # MIN_MATCH_COUNT = 4
    #
    # if len(good)>MIN_MATCH_COUNT:
    #     # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #
    #     src_pts = np.float32([ kp1[m].pt for m in goodIdx ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp2[m].pt for m in goodIdx ]).reshape(-1,1,2)
    #
    # else:
    #     print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    #     matchesMask = None
    #     exit

    return imBoth
