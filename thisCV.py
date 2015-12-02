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
                    q = hierarchy[0, q, 0] # next
                break

    # create mask to delete (not touching the child contours insides)
    mask = np.uint8( np.ones(imgBW.shape) + 254)

    for idx in cTouching:
        col = 0
        cv2.drawContours(mask, contours, idx, col, -1)
    for idx in cInsideTouching:
        col = 255
        cv2.drawContours(mask, contours, idx, col, -1)

    # mask2 = mask.copy()
    # cv2.dilate(mask,mask2)

    cv2.bitwise_and(mask, imgBW, imgBWcopy)
    # imgBWcopy = imgBWcopy * 255
    imgBWcopy = imgBWcopy
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
        # # find normalized_histogram, and its cumulative distribution function
        # hist = cv2.calcHist([im], [0], None, [256], [0, 256])
        # hist_norm = hist.ravel() / hist.max()
        # Q = hist_norm.cumsum()
        #
        # bins = np.arange(256)
        #
        # fn_min = np.inf
        # thresh = -1
        #
        # for i in range(1, 256):
        #     p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        #     q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        #     b1, b2 = np.hsplit(bins, [i])  # weights
        #
        #     # finding means and variances
        #     m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        #     v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
        #
        #     # calculates the minimization function
        #     fn = v1 * q1 + v2 * q2
        #     if fn < fn_min:
        #         fn_min = fn
        #         thresh = i

        # find otsu's threshold value with OpenCV function
        # ret, otsu = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ret, otsu = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
        return otsu

def add_text(im, text, col = 255, hw = (1, 20)):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(im, text, hw , font, 1, 0, 5)
    cv2.putText(im, text, hw, font, 1, col)

class Step():

    def __init__(self, name, function):
        self.name = name
        self.function = function
        self.execution_time_len = 15
        self.execution_time = 0
        self.execution_times = []
        self.mean_execution_time = 0

    def run(self, input):
        start = time.time()
        self.ret = self.function(input)
        end = time.time()
        self.add_exec_times(end-start)
        return self.ret

    def add_exec_times(self, tim):
        if len(self.execution_times) > self.execution_time_len:
            self.execution_times.pop(0)
            self.add_exec_times(tim)
        else:
            self.execution_times.append(tim)
        self.mean_execution_time = np.sum(self.execution_times) / len(self.execution_times)

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
            # im_b = cv2.copyMakeBorder(im[a:-a, a:-a], a, a, a, a,
            #                           cv2.BORDER_CONSTANT, value=color)
            # print(np.max(im_b))
            return cv2.copyMakeBorder(im[a:-a, a:-a], a, a, a, a,
                                      cv2.BORDER_CONSTANT, value=color)

        def make_invert(im):
            return (255 - im)

        def make_flood(im, color = 0):
            # im = im.copy()
            im = im.copy()
            h, w = im.shape[:2]
            a = 2
            mask = np.zeros((h + a, w + a), np.uint8)
            mask[:] = 0
            #seed = None
            seed = (0,0)
            rect = 4
            # rect = 8
            cv2.floodFill(im, mask, seed, color, 0, 255, rect)
            return im

        self.steps.append(Step('gray', make_gray))
        # self.steps.append(Step('clahed', make_clahe))
        # self.steps.append(Step('blurred', make_blur))
        # self.steps.append(Step('gaussed', make_gauss))

        self.steps.append(Step('tresholded', make_otsu))
        self.steps.append(Step('border touch cleared', make_clear_border))
        self.steps.append(Step('removed frame', make_remove_frame))
        self.steps.append(Step('flooded w/white', lambda im: make_flood(im, 255)))
        self.steps.append(Step('flooded w/black', lambda im: make_flood(im, 0)))

    def add_operation(self):
        pass

    def run_all(self, im):
        for step in self.steps:
            im = step.run(im)
        self.ret = im



step_control = StepControl()

def add_operation(operation_name, im_steps, im):
    return im_steps.insert(0, [operation_name, [im]] )

def stepCV(frame, cTag):
    im_steps = []
    a = 0.5
    im = cv2.resize(frame, (0, 0), fx=a, fy=a)

    step_control.run_all(im)
    # for step in step_control.steps:
    #     im = step.function(im)
    #     add_operation( step.name, step_control, im)
    #     # print('hehe',im.shape)

    # ____________________________________________________
    # findTags and put them into im_tags list
    paired, im_tags = findTags(step_control.ret.copy(), cTag)

    return step_control, im_tags

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

