import numpy as np
import cv2

import findHomeography as fh
# from cv2 import xfeatures2d
# import common
import time
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# global variables

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# function definitions


def imclearborder(im, radius, buffer, mask):
    # Given a black and white image, first find all of its contours

    #todo make faster copping as buffer is always the same size!
    buffer = im.copy()
    # _, contours, hierarchy = cv2.findContours(buffer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(buffer, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # _, contours, hierarchy = cv2.findContours(buffer, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    # hierarchy = Next, Previous, First_Child, Parent
    # Get dimensions of image
    n_rows = im.shape[0]
    n_cols = im.shape[1]

    cTouching = []  # indexes of contours that touch the border
    cInsideTouching = [] # indexes that are inside of contours that touch the border

    # print('len contour',len(contours))
    # For each contour...
    for idx in np.arange(len(contours)):
        # if contour is not external continue (otherwise it cannot touoch border
        if hierarchy[0][idx][3] != -1:
            continue

        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= n_rows - 1 - radius and rowCnt < n_rows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= n_cols - 1 - radius and colCnt < n_cols)

            if check1 or check2:

                cTouching.append(idx)

                # add first children inside cInsideTouching
                # = first children is the other edge of external touching contour
                q = hierarchy[0][idx][2] # first child index
                if q != -1:
                    cInsideTouching.append(q)

                # as this contour is already added
                break


    # as the contours are CHAIN_APPROX for speed, this makes also the "noise" dissapear
    approx_correction_thickness = 3

    # create mask to delete (not touching the child contours insides)

    # make the most external contour black
    for idx in cTouching:
        col = 0
        cv2.drawContours(mask, contours, idx, col, thickness=-1)
        cv2.drawContours(mask, contours, idx, col, thickness=approx_correction_thickness )

    # make the inner contours visible
    for idx in cInsideTouching:
        col = 255
        cv2.drawContours(mask, contours, idx, color=col, thickness=-1)
        col = 0
        cv2.drawContours(mask, contours, idx, color=col, thickness=approx_correction_thickness )

    # mask2 = mask.copy()
    # cv2.dilate(mask,mask2)
    cv2.bitwise_and(mask, im, buffer)
    # imgBWcopy = imgBWcopy * 255
    # imgBWcopy = imgBWcopy
    return buffer



def extractContourArea(im_scene, external_contour):
    mask = np.uint8( np.zeros(im_scene.shape) )
    col = 1
    cv2.drawContours(mask, [external_contour], 0, col, -1)

    scene_with_tag = np.uint8( np.zeros(im_scene.shape) )
    cv2.bitwise_and(mask, im_scene.copy(), scene_with_tag)
    scene_with_tag = scene_with_tag * 255

    return scene_with_tag


def findTagsInScene(im_scene, model_tag):
    allowed_size = 800
    max_size = max(im_scene.shape)
    if max_size > allowed_size:
        # find tags in smaller image
        a = allowed_size/max_size
        im_scene_smaller = cv2.resize(im_scene,(0,0),fx=a,fy=a)
        markuped_scene, seen_tags = findTags(im_scene_smaller, model_tag)
        if seen_tags is None or len(seen_tags) == 0:
            markuped_scene, seen_tags = findTags(im_scene, model_tag)
    else:
        markuped_scene, seen_tags = findTags(im_scene, model_tag)

    return markuped_scene, seen_tags

def findTags(im_scene, model_tag):

    # first create copy of scene (not to be contoured)
    scene_markuped = im_scene.copy()


    # _, contours, hierarchy = cv2.findContours(im_scene.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    _, external_contours, hierarchy = cv2.findContours(im_scene.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    # _, external_contours, hierarchy = cv2.findContours(im_scene.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    # _, contours, hierarchy = cv2.findContours(im_scene.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )

    seen_tags = []

    # for every external contour area
    for external_contour in external_contours:

        # zero out everything but the tag area
        scene_with_tag = extractContourArea(im_scene, external_contour)

        # initialize observed tag
        observed_tag = fh.C_observedTag(scene_with_tag, external_contour, scene_markuped)


        # find out if the tag is in the area
        observed_tag.calculate(model_tag)

        # append it to seen tags list
        seen_tags.append(observed_tag)

    return scene_markuped, seen_tags

    # # for every external contour area
    # for q in np.arange(len(external_contours)):
    #
    #     # leave only the biggest contour
    #     # ?? - gets rid of the "noise"
    #
    #
    #     # # remove all inner contours
    #     # if hierarchy[0][q][3] != -1:
    #     #     continue
    #
    #     # zero out everything but the tag
    #     mask = np.uint8( np.zeros(im_scene.shape) )
    #     col = 1
    #     cv2.drawContours(mask, external_contours, q, col, -1)
    #
    #     scene_with_tag = np.uint8( np.zeros(im_scene.shape) )
    #     cv2.bitwise_and(mask, im_scene.copy(), scene_with_tag)
    #     scene_with_tag = scene_with_tag * 255
    #     scene_with_tag = extractContourArea(im_scene, external_contour)
    #
    #     # # find out euler number
    #     # _, tagCnt, hie = cv2.findContours(scene_with_tag.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     # if hie is None: continue
    #     # if len(hie) == 0: continue
    #     # hie = hie[0]
    #     # innerSquaresCount = 2
    #     # cntNumMin = 2  # case of joined inner squares
    #     # cntNumMax = 1 + innerSquaresCount
    #
    #     # if len(hie) >= cntNumMin and len(hie) <= cntNumMax:
    #     #     imTags.append(scene_with_tag)
    #     #     # imTags = imIn[y:y+h,x:x+w]
    #
    #     # if it sustains tha check of some kind
    #
    #     observed_tag = fh.C_observedTag(scene_with_tag, scene_markuped)
    #     # observed_tag = fh.C_observedTag(scene_with_tag, None)
    #     observed_tag.calculate(model_tag)
    #
    #     seen_tags.append(observed_tag)
    #
    # return scene_markuped, seen_tags

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
        ret, otsu = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu
    elif type == 'otsu_inv':
        ret, otsu = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

    def str_mean_execution_time(self):
        return '{0:.2f} ms'.format(round(self.mean_execution_time * 1000,2))

class StepControl():

    buffer = None
    seen_tags = None
    mask_ones = None

    def recreate_mask(self,im ):
        if self.mask_ones is None or im.shape != self.mask_ones.shape:
            self.mask_ones = np.uint8( np.ones(im.shape) + 254)
            # print('recreated mask')

    def get_mask(self, im):
        self.recreate_mask(im)
        return self.mask_ones.copy()


    def recreate_buffer(self, im):
        if self.buffer is None:
            self.buffer = im.copy()
        else:
            if im.shape != self.buffer.shape:
                self.buffer = im.copy()
            # else:
            #     return self.buffer


    def get_buffer(self, im):
        if self.buffer is None:
            self.recreate_buffer(im)
        return self.buffer


    def add_available_step(self, name, function):
        self.available_steps[name] = Step(name, function)

    def define_available_steps(self):

        self.available_steps = {}

        def make_nothing(im):
            return im

        def make_resize(im):
            return cv2.resize(im.copy(), (0, 0), fx=self.resolution_multiplier, fy=self.resolution_multiplier)

        def make_gray(im):
            return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        def make_clahe(im):
            return clahe.apply(im)

        def make_blur(im, a=75):
            return cv2.bilateralFilter(im, 9, a, a)

        def make_gauss(im, a=5, sigma=1):
            return cv2.GaussianBlur(im.copy(), (a, a), sigmaX=sigma)

        def make_median(im, a=5):
            return cv2.medianBlur(im, a)

        def make_otsu(im):
            return threshIT(im,'otsu').copy()

        def make_otsu_inv(im):
            return threshIT(im,'otsu_inv').copy()


        def make_clear_border(im, width = 5):
            return imclearborder(im, width, self.get_buffer(im), self.get_mask(im))

        def make_remove_frame(im, width = 5, color = 0):
            a = width
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


        # Initiate ORB detector
        scoreType = cv2.ORB_FAST_SCORE
        # scoreType = cv2.ORB_HARRIS_SCORE
        orb = cv2.ORB_create(scoreType=scoreType)
        orb.setMaxFeatures(300)
# 'setEdgeThreshold','setFastThreshold', \
# 'setFirstLevel', 'setMaxFeatures', \
# 'setNLevels', 'setPatchSize', \
# 'setScaleFactor', 'setScoreType',
#         setWTA_K

        # orb = cv2.ORB_create()

        def make_orb(im):
            # find the keypoints with ORB
            kp = orb.detect(im, None)

            # compute the descriptors with ORB
            kp, des = orb.compute(im, kp)

            # draw only keypoints location,not size and orientation
            # col = (0,255,0)
            col = 142
            im_out = np.zeros(im.shape)
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + (
                    cv2.DRAW_MATCHES_FLAGS_DEFAULT)

            cv2.drawKeypoints(im, kp, im_out, color=col, flags=flags)
            print(len(kp))
            return im_out


        # def make_dif_laplace(im):



        # Init SURF detector
        surf = cv2.xfeatures2d.SURF_create()

        def make_surf(im):
            kp = surf.detect(im, None)

            col = 142
            im_out = np.zeros(im.shape)
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + (
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.drawKeypoints(im, kp, im_out, color=col, flags=flags)

            return im_out

        # Init SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        def make_sift(im):
            kp = sift.detect(im, None)

            col = 142
            im_out = np.zeros(im.shape)
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + (
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.drawKeypoints(im, kp, im_out, color=col, flags=flags)

            return im_out

        freakExtractor = cv2.xfeatures2d.FREAK_create()
        def make_freak(im):
            keypoints, descriptors = freakExtractor.compute(im, keypoints)

        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create(nonmaxSuppression=0) # 8ms
        fast = cv2.FastFeatureDetector_create(nonmaxSuppression=1) # 3ms
        # fast.setBool('nonmaxSuppression',1)


        def make_fast(im):

            # find and draw the keypoints
            kp = fast.detect(im, None)

            col = 255
            im_out = np.zeros(im.shape)
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + (
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


            keypoints, descriptors = freakExtractor.compute(im, kp)

            cv2.drawKeypoints(im, kp, im_out, color=col, flags=flags)


            return im_out


        def make_hls_stack(im):
            im_gray = make_gray(im)
            im_hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS_FULL)
            h,l,s = cv2.split(im_hls)
            im1 = fh.joinIm([[h],[l]])


            im2 = fh.joinIm([[s], [im_gray]])
            # print(im1.shape,'ass',im2.shape)
            im_out = fh.joinIm([[im1], [im2]], vertically=1)
            # print(im_out.shape)
            return im_out

        def make_hls_saturation(im):
            # im_hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
            im_hls = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
            # return np.uint8( im_hls[:, :, 2] )
            return np.uint8( im_hls[:, :, 1] )

        def make_erase_color(im):
            im_gray = make_gray(im)
            gray_field = np.uint8( np.ones(im_gray.shape) + 0)
            # np.uint8( np.ones(im.shape) + 254)

            im_out = im_gray.copy()

            mask =  make_gauss(make_hls_saturation(im), a =9, sigma=13)

            # _, th1 = cv2.threshold(mask, 127, 255, cv2.THRESH_TOZERO)
            # mask = th1

            # mask = np.round(mask)
            cv2.bitwise_and(mask=mask, src1=im_gray, src2=gray_field, dst=im_out)
            return mask
            # return gray_field
            # return im_gray
            return im_out

        self.add_available_step('original', make_nothing)
        self.add_available_step('gray', make_gray)
        self.add_available_step('clahed', make_clahe)
        self.add_available_step('blurred', make_blur)
        self.add_available_step('gaussed', make_gauss)

        self.add_available_step('median', make_median)

        self.add_available_step('resize', make_resize)
        self.add_available_step('invert', make_invert)

        self.add_available_step('tresholded', make_otsu)
        self.add_available_step('tresholded inverted', make_otsu_inv)
        self.add_available_step('border touch cleared', make_clear_border)
        self.add_available_step('removed frame', make_remove_frame)
        self.add_available_step('flooded w/white', lambda im: make_flood(im, 255))
        self.add_available_step('flooded w/black', lambda im: make_flood(im, 0))

        self.add_available_step('hls stack', make_hls_stack)
        self.add_available_step('hls saturation', make_hls_saturation)
        self.add_available_step('erase color', make_erase_color)


        self.add_available_step('orb', make_orb)
        self.add_available_step('sift', make_sift)
        self.add_available_step('surf', make_surf)

        self.add_available_step('freak', make_freak)
        self.add_available_step('fast', make_fast)

        # self.available_steps.append(Step('original', make_nothing))
        # self.available_steps.append(Step('gray', make_gray))
        # self.steps.append(Step('clahed', make_clahe))
        # self.steps.append(Step('blurred', make_blur))
        # self.steps.append(Step('gaussed', make_gauss))
        #
        # self.available_steps.append(Step('resize', make_resize))
        #
        # self.available_steps.append(Step('tresholded', make_otsu))
        # self.available_steps.append(Step('border touch cleared', make_clear_border))
        # self.available_steps.append(Step('removed frame', make_remove_frame))
        # self.available_steps.append(Step('flooded w/white', lambda im: make_flood(im, 255)))
        # self.available_steps.append(Step('flooded w/black', lambda im: make_flood(im, 0)))


    def select_steps(self, model_tag):
        self.model_tag = model_tag
        def make_find_tags(im):
            markuped_scene, seen_tags = findTagsInScene(im.copy(), self.model_tag)
            self.seen_tags = seen_tags
            return markuped_scene

        # add find tag with currentyl selected model_tag
        findtag_name = 'findTags'
        self.available_steps.pop(findtag_name, None)
        self.add_available_step(findtag_name, make_find_tags)

        # create steps list for this model_tag algorithm
        self.steps = []
        [self.steps.append(self.available_steps[step_name]) for step_name in self.model_tag.step_names]


    def __init__(self, resolution_multiplier, model_tag):

        self.resolution_multiplier = resolution_multiplier
        self.define_available_steps()
        self.select_steps(model_tag)


    def add_operation(self):
        pass

    def run_all(self, im):
        for step in self.steps:
            im = step.run(im)
        self.ret = im

    def step_all(self, im, resolution_multiplier):
        self.resolution_multiplier = resolution_multiplier
        self.run_all(im)


def add_operation(operation_name, im_steps, im):
    return im_steps.insert(0, [operation_name, [im]] )

# def loopCV(cap):
#     print("loopCV started")
#     while (True):
#         im = stepCV(cap)
#         cv2.imshow('image', im)
#         # How to end the loop
#         k = cv2.waitKey(30) & 0xff
#         if k == ord('q'):
#             break
#         if k == 27:
#             break
#         cv2.destroyAllWindows() # When everything done, release the capture

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
    # loopCV(cap)
    cap.release()
        # cnt = external_contours[q]

    cv2.destroyAllWindows()

