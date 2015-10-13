import numpy as np
import cv2
import findHomeography as fh
# from cv2 import xfeatures2d
# import common
# from plane_tracker import PlaneTracker

from collections import Counter


global tracker, ar_verts, ar_edges
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### imclearborder definition

def imclearborder(imgBW, radius):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # hierarchy = Next, Previous, First_Child, Parent
    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]

    cTouching = []  # indexes of contours that touch the border
    cInsideTouching = [] # indexes that are inside of contours that touch the border
    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]
        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows - 1 - radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols - 1 - radius and colCnt < imgCols)

            if check1 or check2:
                cTouching.append(idx)
                # add children inside cInsideTouching
                q = hierarchy[0][idx][2] # first child index
                while q != -1:
                    cInsideTouching.append(q)
                    q = hierarchy[0,q,0] # next
                break

    # create mask to delete (not touching the child contours insides)
    mask = np.uint8( np.ones(imgBW.shape) )
    for idx in cTouching:
        col = 0
        cv2.drawContours(mask, contours, idx, col, -1)
    for idx in cInsideTouching:
        col = 1
        cv2.drawContours(mask, contours, idx, col, -1)

    # mask2 = mask.copy()
    # cv2.dilate(mask,mask2)

    cv2.bitwise_and(mask, imgBW, imgBWcopy)
    imgBWcopy = imgBWcopy * 255
    return imgBWcopy


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### makepairs(im)

def draw_overlay(vis, tracked):
    x0, y0, x1, y1 = tracked.target.rect
    quad_3d = np.float32([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
    fx = 0.5 + cv2.getTrackbarPos('focal', 'plane') / 50.0
    h, w = vis.shape[:2]
    K = np.float64([[fx * w, 0, 0.5 * (w - 1)],
                    [0, fx * w, 0.5 * (h - 1)],
                    [0.0, 0.0, 1.0]])
    dist_coef = np.zeros(4)
    ret, rvec, tvec = cv2.solvePnP(quad_3d, tracked.quad, K, dist_coef)
    verts = ar_verts * [(x1 - x0), (y1 - y0), -(x1 - x0) * 0.3] + (x0, y0, 0)
    verts = cv2.projectPoints(verts, rvec, tvec, K, dist_coef)[0].reshape(-1, 2)
    for i, j in ar_edges:
        (x0, y0), (x1, y1) = verts[i], verts[j]
        cv2.line(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 2)
    return vis


def findTags(imScene, cTagModel):
    # Given a black and white image, first find all of its contours
    # _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    _, contours, hierarchy = cv2.findContours(imScene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    # contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
    imTags = [];

    cSeenTags = []
    imSceneWithDots = imScene.copy()
    # find bounding boxes etc
    for q in np.arange(len(contours)):
        cnt = contours[q]

        # moments
        mu = cv2.moments(cnt)
        # leftTop = tuple(cnt[0, 0])
        if mu['m00'] == 0: continue
        mc = (int(mu['m10'] / mu['m00']), int(mu['m01'] / mu['m00']))

        # DRAW centroid
        color = 200
        cv2.circle(imSceneWithDots, mc, 4, color, -1, 8, 0)

        # non-rotated boundingbox
        x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)
        # bounding box slice
        # imTagInScene = imIn[y:y + h, x:x + w]

        # slice out imTagInScene
        mask = np.uint8( np.zeros(imScene.shape) )
        col = 1
        cv2.drawContours(mask, contours, q, col, -1)

        imTagInScene = np.uint8( np.zeros(imScene.shape) )
        cv2.bitwise_and(mask, imScene.copy(), imTagInScene)
        imTagInScene = imTagInScene * 255

        # # find out euler number
        # _, tagCnt, hie = cv2.findContours(imTagInScene.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # if hie is None: continue
        # if len(hie) == 0: continue
        # hie = hie[0]
        # innerSquaresCount = 2
        # cntNumMin = 2  # case of joined inner squares
        # cntNumMax = 1 + innerSquaresCount

        # if len(hie) >= cntNumMin and len(hie) <= cntNumMax:
        #     imTags.append(imTagInScene)
        #     # imTags = imIn[y:y+h,x:x+w]

        # if it sustains tha check of some kind
        if 1:
            cSeenTag = fh.C_observedTag(imTagInScene)
            # cSeenTag.findWarpMatrix()
            # mWarp = cSeenTag.mWarp
            #
            dst_pts, mWarp = fh.findWarpMatrix(imTagInScene, cTagModel)

            # get inverse transformation matrix
            try:
                mInverse = np.linalg.inv(mWarp)
                imTagRecreated = fh.drawSceneWarpedToTag(mInverse, imTagInScene, cTagModel.imTagDetect.shape)
                imTags.append(imTagRecreated )
            except:
                print "Probably bad tag detected"
                continue
            cSeenTags.append(cSeenTag)

    return imSceneWithDots, imTags


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### bwareaopen definition
def bwareaopen(imgBW, areaPixels,col = 0):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, col, -1)
    return imgBWcopy


def update(i):
    tbValue = i
    print(i)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def inverte(im):
    return (255 - im)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def threshIT(im, type):
    ## THRESH
    if type == cv2.THRESH_BINARY_INV or type == 1:
        _, th1 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
        return th1
    elif type == cv2.ADAPTIVE_THRESH_MEAN_C or type == 2:
        th2 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        return th2
    elif type == cv2.ADAPTIVE_THRESH_MEAN_C or type == 3:
        th3 = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        return th3

    ## OTSU
    elif type == 'otsu' or type == 4:
        # find normalized_histogram, and its cumulative distribution function
        hist = cv2.calcHist([im], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.max()
        Q = hist_norm.cumsum()

        bins = np.arange(256)

        fn_min = np.inf
        thresh = -1

        for i in range(1, 256):
            p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
            q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
            b1, b2 = np.hsplit(bins, [i])  # weights

            # finding means and variances
            m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
            v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2

            # calculates the minimization function
            fn = v1 * q1 + v2 * q2
            if fn < fn_min:
                fn_min = fn
                thresh = i

        # find otsu's threshold value with OpenCV function
        # ret, otsu = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ret, otsu = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
        return otsu

def gaussIt(im,a):
    return cv2.GaussianBlur(im, (a, a), 0)

def floodIt(im,newVal):
    h, w = im.shape[:2]
    # mask = np.zeros((h + 2, w + 2), np.uint8)
    a = 2
    mask = np.zeros((h + a, w + a), np.uint8)
    mask[:] = 0
    # seed = None
    seed = (0,0)
    rect = 4
    # rect = 8
    cv2.floodFill(im, mask, seed, newVal, 0, 255, rect)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def stepCV(frame, cTag):
    a = 0.5
    im = cv2.resize(frame, (0, 0), fx=a, fy=a)
    # im = frame

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = gray
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # adaptive image histogram equalization
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(im)
    im = cl1
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## gaussian blur
    blur = gaussIt(im,7)
    im = blur
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##
    im = inverte(im.copy())
    thresh = threshIT(im, 'otsu')
    im = thresh
    im = inverte(im.copy())
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # inversion
    # im = inverte(im.copy())
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # imclearborder
    # maks out all contours which are touching the border
    killerBorder = 5
    killedBorder= imclearborder(im, killerBorder)
    a = 1
    killedBorder = cv2.copyMakeBorder(killedBorder[a:-a, a:-a], a, a, a, a, cv2.BORDER_CONSTANT, value=0)

    im = killedBorder
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # bwareaopen
    # delete too small groups of pixels - with contours - slow
    # col = 64
    # opened = bwareaopen(im, 5*5, col)
    # im = opened

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # imfill
    # flood
    flooded = im.copy()
    floodIt(flooded, 255)
    floodIt(flooded, 0)
    im = flooded
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # findTags and put them into imTags list

    paired, imTags = findTags(im, cTag)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # track individual tags orientation
    # vis = []
    # for imTag in imTags:
    #     a = 50
    #
    #     # make border for better drawing
    #     im = cv2.copyMakeBorder(imTag, a, a, a, a, cv2.BORDER_CONSTANT, value=0)
    #     vis.append(im)
    #
    # imTags = vis
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # show of

    ims = []
    # ims.append( [gray] )
    ims.append( [cl1] )
    # ims.append( [blur] )
    ims.append( [thresh] )
    ims.append( [killedBorder] )
    ims.append( [flooded] )
    ims.append( [paired] )

    imWhole = fh.joinIm(ims, 1)
    # imWhole = np.vstack(ims)

    # if len(imTags) > 0:
    #     imTags = imTags[0]
    #    cv2.imshow('imColor', frame)
    return imWhole, imTags


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def loopCV(cap):
    print("loopCV started")
    while (True):
        im = stepCV(cap)
        cv2.imshow('image', im)

        # End loop
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        if k == 27:
            break

        # When everything done, release the capture
        cv2.destroyAllWindows()

def waitKeyExit():
    while True:
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        if k == 27:
            break
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### Main program

# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    ims = []

    cv2.namedWindow('image')

    tbValue = 3
    maxValue = 6
    cv2.createTrackbar("trackMe", "image", tbValue, maxValue, update)
    loopCV(cap)
    cap.release()
    cv2.destroyAllWindows()

