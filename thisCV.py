import numpy as np
import cv2
import findHomeography as fh
# from cv2 import xfeatures2d
# import common
import time
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# global variables
global tracker, ar_verts, ar_edges


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# function definitions

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

def findTags(imScene, cTagModel):

    # _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    _, contours, hierarchy = cv2.findContours(imScene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    imTags = [];

    cSeenTags = []
    imSceneWithDots = imScene.copy()

    # find bounding boxes etc
    for q in np.arange(len(contours)):

        cnt = contours[q]


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

            if cSeenTag.addExternalContour(cnt) != 0:
                continue
            if cSeenTag.findWarpMatrix(cTagModel) != 0:
                continue

            imTagRecreated = cSeenTag.drawSceneWarpedToTag(cTagModel)
            fh.drawCentroid(imSceneWithDots, cnt, 180) # DRAW centroid

            imTags.append(imTagRecreated )
            cSeenTags.append(cSeenTag)

    return imSceneWithDots, imTags

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

# def inverte(im):
#     return (255 - im)

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

# def gaussIt(im,a):
#     return cv2.GaussianBlur(im, (a, a), 0)
#
# def blurIt(im,a):
# #     a = 75
#     return cv2.bilateralFilter(im,9,a,a)
#
# def flood_it(im,newVal):
#     h, w = im.shape[:2]
#     # mask = np.zeros((h + 2, w + 2), np.uint8)
#     a = 2
#     mask = np.zeros((h + a, w + a), np.uint8)
#     mask[:] = 0
#     # seed = None
#     seed = (0,0)
#     rect = 4
#     # rect = 8
#     cv2.floodFill(im, mask, seed, newVal, 0, 255, rect)

def add_operation(operation_name, im_steps, im):
    return im_steps.insert(0, [operation_name, [im] ] )

class Step():

    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.execution_time_len = 15
        self.execution_time = 0
        self.mean_execution_time = 0

    def run(self, input):
        start = time.time()
        self.ret = self.function(input)
        end = time.time()
        self.add_exec_times(end-start)
        return self.ret

    def add_exec_times(self, tim):
        if len(self.execution_time) > self.execution_time_len:
            self.execution_time.pop(0)
            self.add_exec_times(tim)
        else:
            self.execution_time.append(tim)
        self.mean_execution_time = np.sum(self.execution_time) / len(self.execution_time)

class StepControl():

    def __init__(self):
        self.steps = []

        def make_gray(im):
            return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        def make_clahe(im):
            return clahe.apply(im)

        a = 75
        def make_blur(im, a=75):
            return cv2.bilateralFilter(im,9,a,a)
        a = 5
        def make_gauss(im, a=5):
            return cv2.GaussianBlur(im, (a, a), 0)

        def make_otsu(im):
            return threshIT(im,'otsu').copy()

        def make_clear_border(im, width = 5):
            return imclearborder(im, width)

        def make_remove_frame(im, width = 5, color = 0):
            return cv2.copyMakeBorder(im[a:-a, a:-a], a, a, a, a,
                                      cv2.BORDER_CONSTANT, value=color)

        def make_invert(im):
            return (255 - im)

        def make_flood(im, color = 0):
            h, w = im.shape[:2]
            a = 2
            mask = np.zeros((h + a, w + a), np.uint8)
            mask[:] = 0
            # seed = None
            seed = (0,0)
            rect = 4
            # rect = 8
            cv2.floodFill(im, mask, seed, color, 0, 255, rect)
            return im

        self.steps.append(Step('gray', make_gray))
        self.steps.append(Step('clahed', make_clahe))
        # self.steps.append(Step('blurred', make_blur))
        # self.steps.append(Step('gaussed', make_gauss))

        self.steps.append(Step('tresholded', make_otsu))
        self.steps.append(Step('border touch cleared', make_clear_border))
        self.steps.append(Step('removed frame', make_remove_frame))
        self.steps.append(Step('flooded w/white', lambda im: make_flood(im, 255)))
        self.steps.append(Step('flooded w/black', lambda im: make_flood(im, 0)))
#
# flooded = im.copy()
#         flood_it(flooded, 255)
#     im = flooded
#     add_operation( 'flooded with white', im_steps, im)
#
#     flood_it(flooded, 0)
#     im = flooded
#     add_operation( 'flooded with black', im_steps, im)
#
step_control = StepControl()

def stepCV(frame, cTag):
    im_steps = []
    a = 0.5
    im = cv2.resize(frame, (0, 0), fx=a, fy=a)
    # im = frame
    # add_operation( 'resized', im_steps, im)
    # ____________________________________________________
    # RGB -> gray
    for step in step_control.steps:
        im = step.function(im)
        add_operation( step.name, im_steps, im)
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im = gray
    # add_operation( 'gray', im_steps, im)
    # ____________________________________________________
    # adaptive image histogram equalization - create a CLAHE object (Arguments are optional).
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    # cl1 = clahe.apply(im)
    # im = cl1
    # add_operation( 'clahed', im_steps, im)
    # ____________________________________________________
    # gaussian blur
    # blur = gaussIt(im,7)
    # blur = blurIt(im,75)
    # im = blur
    # imDict.update( {'blurred',im} )
    # ____________________________________________________
    # Threshing - to get binary image out of gray one
    # im = inverte(im.copy())
    # add_operation( 'inverted', im_steps, im)
    # thresh = threshIT(im, 'otsu')
    # im = thresh
    # add_operation( 'thresholded', im_steps, im)
    # im = inverte(im.copy())
    # add_operation( 'inverted again', im_steps, im)
    # ____________________________________________________
    # inversion
    # im = inverte(im.copy())
    # ____________________________________________________
    # imclearborder - maskes out all contours which are touching the border
    # killer_border_width = 5
    # border_touch_cleared = imclearborder(im, killer_border_width)
    # im = border_touch_cleared
    # add_operation( 'border touch cleared', im_steps, im)
    # a = 1
    # removed_frame = cv2.copyMakeBorder(im[a:-a, a:-a], a, a, a, a, cv2.BORDER_CONSTANT, value=0)
    #
    # im = removed_frame
    # add_operation( 'removed frame', im_steps, im)
    # ____________________________________________________
    # bwareaopen
    # delete too small groups of pixels - with contours - slow
    # col = 64
    # opened = bwareaopen(im, 5*5, col)
    # im = opened

    # ____________________________________________________
    # imfill
    # flood
    # flooded = im.copy()
    # flood_it(flooded, 255)
    # im = flooded
    # add_operation( 'flooded with white', im_steps, im)
    #
    # flood_it(flooded, 0)
    # im = flooded
    # add_operation( 'flooded with black', im_steps, im)
    # ____________________________________________________
    # findTags and put them into im_tags list
    paired, im_tags = findTags(im, cTag)

    # ____________________________________________________
    # create progress image
    # ims = []
    # # ims.append( [gray] )
    # ims.append( [cl1] )
    # # ims.append( [blur] )
    # ims.append( [thresh] )
    # ims.append( [killed_border] )
    # ims.append( [flooded] )
    # ims.append( [paired] )
    # #
    # imWhole = fh.joinIm(ims, 1)
    # ____________________________________________________
    # FPS
    # font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    # col = 255
    # text = 'FPS = ?'
    # hw = (1,20)
    # cv2.putText(imWhole, text, hw , font, 1, 0, 5) # , cv2.LINE_AA )
    # cv2.putText(imWhole, text, hw, font, 1, col)
    # ____________________________________________________
    # return imWhole, im_tags
    return im_steps, im_tags
    # return im_steps, im_tags

def loopCV(cap):
    print("loopCV started")
    while (True):
        im = stepCV(cap)
        cv2.imshow('image', im)
        # How to end the loop
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        if k == 27:
            break
        cv2.destroyAllWindows() # When everything done, release the capture

def waitKeyExit():
    while True:
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        if k == 27:
            break

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    ims = []

    cv2.namedWindow('image')

    tbValue = 3
    maxValue = 6
    # cv2.createTrackbar("trackMe", "image", tbValue, maxValue, update)
    loopCV(cap)
    cap.release()
    cv2.destroyAllWindows()

