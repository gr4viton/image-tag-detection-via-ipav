import numpy as np
import cv2
# from cv2 import xfeatures2d
# import common
from plane_tracker import PlaneTracker

from collections import Counter


global tracker, ar_verts, ar_edges
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### imclearborder definition

def imclearborder(imgBW, radius):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]

    contourList = []  # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]
        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows - 1 - radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols - 1 - radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        color = 0
        cv2.drawContours(imgBWcopy, contours, idx, color, -1)

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


def findTags(imIn):
    # Given a black and white image, first find all of its contours
    im = imIn.copy()
    # _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )

    _, contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]

    # # Get dimensions of image
    # imgRows = imIn.shape[0]
    # imgCols = imIn.shape[1]

    # contourList = [] # list of pair-lists

    # imTags = []
    imTags = [];
    #  For each contour...
    # sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(im, None)

    # find bounding boxes etc
    for q in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[q]
        # moments
        mu = cv2.moments(cnt)
        leftTop = tuple(cnt[0, 0])
        if mu['m00'] == 0: continue
        mc = (int(mu['m10'] / mu['m00']), int(mu['m01'] / mu['m00']))
        # first pixel
        color = 128
        cv2.circle(im, leftTop, 4, color, -1, 8, 0)
        # centroid
        color = 200
        cv2.circle(im, mc, 4, color, -1, 8, 0)

        # non-rotated boundingbox
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)

        # rotated boundingbox
        color = 122
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # mateches
        # im = cv2.drawContours(im,[box],0,color,2)
        # im = cv2.drawKeypoints(im ,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        imTag = imIn[y:y + h, x:x + w]
        _, tagCnt, hie = cv2.findContours(imTag.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hie == None: continue
        if len(hie) == 0: continue
        hie = hie[0]
        innerSquaresCount = 2
        cntNumMin = 2  # case of joined inner squares
        cntNumMax = 2 + innerSquaresCount

        if len(hie) >= cntNumMin and len(hie) <= cntNumMax:
            imTags.append(imTag)
            # imTags = imIn[y:y+h,x:x+w]

    return im, imTags


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            color = 64
            cv2.drawContours(imgBWcopy, contours, idx, color, -1)
    return imgBWcopy


def update(i):
    tbValue = i
    print i


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def inverte(im):
    return (255 - im)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def threshIT(im, type):
    ## THRESH
    if type == cv2.THRESH_BINARY_INV or type == 1:
        ret, th1 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
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

        for i in xrange(1, 256):
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
        ret, otsu = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return otsu


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def stepCV(cap):
    flag, frame = cap.read()
    if flag == 0:
        return None
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
    a = 5
    blur = cv2.GaussianBlur(im, (a, a), 0)
    im = blur
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ##
    im = threshIT(blur, 'otsu')
    thresh = im
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # border filler
    killerBorder = 5
    clear = imclearborder(im, killerBorder)
    a = 5
    clear = cv2.copyMakeBorder(clear[a:-a, a:-a], a, a, a, a, cv2.BORDER_CONSTANT, value=0)
    # cv2.findContours(otsu

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # bwareaopen
    im = clear
    opened = bwareaopen(im, 101)
    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # imfill
    im = clear

    flooded = im.copy()

    h, w = im.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    mask[:] = 0
    seed = None

    cv2.floodFill(flooded, mask, seed, 0, 0, 255, 4)

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # findTags and put them into imTags list
    im = flooded
    paired, imTags = findTags(im)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # track individual tags orientation
    vis = []
    for imTag in imTags:
        a = 50
        # make a edge image
        # edgeTag =  np.uint8(np.absolute( cv2.Sobel(imTag, cv2.CV_64F, 1, 0, ksize=5) ))
        # Laplacian
        edgeTag = np.uint8(np.absolute(cv2.Laplacian(imTag, cv2.CV_64F)))
        edgeTag = imTag

        # make border for better drawing
        im = cv2.copyMakeBorder(edgeTag, a, a, a, a, cv2.BORDER_CONSTANT, value=0)
        vis.append(im)
        #
        # tracker = getTracker()
        # tracked = tracker.track(edgeTag)
        # color = 65
        # for tr in tracked:
        #     cv2.polylines(vis, [np.int32(tr.quad)], True, color, 2)
        #     for (x, y) in np.int32(tr.p1):
        #         cv2.circle(vis, (x, y), 2, color)
        #     draw_overlay(vis, tr)
        #     print "drawn Contours"
            # H, status = cv2.findHomography(p0, p1, cv2., 3.0) # cv2.RANSAC not needed

    imTags = vis
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # virtual overlay
    # imTags = findRotation(imTags)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # show of

    # ims = [gray, cl1, blur, thresh, clear, flooded]
    ims = [gray, cl1, clear, paired]
    imWhole = np.vstack(ims)

    # if len(imTags) > 0:
    #     imTags = imTags[0]
    #    cv2.imshow('imColor', frame)
    return imWhole, imTags


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def findRotation(imTags):
    # print len(imTags)
    # if len(imTags) == 0:
    #     return
    im = imTags
    # for im in imTags:
    tracked = tracker.track(im)
    # tracked = []
    for tr in tracked:
        cv2.polylines(im, [np.int32(tr.quad)], True, (255, 255, 255), 2)
        for (x, y) in np.int32(tr.p1):
            cv2.circle(im, (x, y), 2, (255, 255, 255))
        im = draw_overlay(im, tr)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def loopCV(cap):
    print "loopCV started"
    while (True):
        im = stepCV(cap)
        cv2.imshow('image', im)
        # if __name__ == '__main__':
        #     cv2.imshow('image', im )
        # else:
        #     print "returning im"
        #     return im

        # End loop
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        if k == 27:
            break

            # When everything done, release the capture


        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #### Main program

        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding

        # C:\PROG\dev\opencv\opencv\sources\data\haarcascades\


def getTracker():
    #global tracker, ar_verts, ar_edges
    tracker = PlaneTracker()
    tag = cv2.imread('tag1.png', 0)
    # print(tag.shape)
    rect = [0, 0, tag.shape[0], tag.shape[1]]

    a = max(tag.shape)
    preparedTag = np.uint8(np.absolute(cv2.Laplacian(tag, cv2.CV_64F)))
    preparedTag = cv2.copyMakeBorder(preparedTag, a, a, a, a, cv2.BORDER_CONSTANT, value=0)
    tracker.add_target(preparedTag, rect)
    print("TAG tracker added here")
    ar_verts = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                           [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1],
                           [0, 0.5, 2], [1, 0.5, 2]])
    ar_edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
                (4, 8), (5, 8), (6, 9), (7, 9), (8, 9)]
    return tracker

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