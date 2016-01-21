import numpy as np
import cv2
from enum import Enum

import matplotlib.pyplot as plt
from itertools import cycle
import howToFindSqure as sq

class Error(Enum):
    flawless = 0
    no_square_points = 1
    no_inverse_matrix = 2
    no_tag_rotations_found = 3
    rotation_uncertainity = 4
    external_contour_error = 5
    contour_too_small = 6
    square_points_float_nan = 7

import time

class C_observedTag:
    # static class variable - known Tag
    # tagModels = loadTagModels('2L')

    def __init__(self, imTagInScene, external_contour, scene_markuped):
        self.imScene = imTagInScene # image of scene in which the tag is supposed to be
        self.imWarped = None # ground floor image of tag transformed from imScene
        # self.tag_warped = None # warped tag image into model_tag space

        self.dst_pts = None # perspectively deformed detectionArea square corner points
        self.mWarp2tag = None # transformation matrix from perspective scene to ground floor tag
        self.mWarp2scene = None # transformation matrix from ground floor tag to perspective scene
        self.cntExternal = external_contour # detectionArea external contour
        self.mu = None # all image moments
        self.mc = None # central moment
        self.rotation = None # square symbols in symbolArea check possible rotations similar to cTagModel - [0,90,180,270]deg
        self.error = Error.flawless # error
        self.scene_markuped = scene_markuped # whole image to print additive markups of this observed tag to

        self.color_corners = 160
        self.color_centroid = 180

        # self.external_contour_approx = cv2.CHAIN_APPROX_SIMPLE
        # self.external_contour_approx = cv2.CHAIN_APPROX_NONE
        # self.external_contour_approx = cv2.CHAIN_APPROX_TC89_L1
        self.external_contour_approx = cv2.CHAIN_APPROX_TC89_KCOS

        self.verbatim = False # if set to True the set_error function would print findTag error messages in default output stream

        self.minimum_contour_length = 4

    def calcMoments(self):  # returns 0 on success
        self.mu = cv2.moments(self.cntExternal)
        if self.mu['m00'] == 0:
            return 1
        self.mc = np.float32(getCentralMoment(self.mu))
        return 0

    def calcExternalContour(self): # returns 0 on success
        _, contours, hierarchy = cv2.findContours(self.imScene.copy(), cv2.RETR_EXTERNAL, self.external_contour_approx )
        if len(contours) != 0:
            self.cntExternal = contours[0]
        return self.calcMoments()

    def getExternalContour(self, imScene):
        _, contours, hierarchy = cv2.findContours(imScene.copy(), cv2.RETR_EXTERNAL, self.external_contour_approx )
        return contours[0]

    def addExternalContour(self, cntExternal): # returns 0 on success
        self.cntExternal = cntExternal
        return self.calcMoments()

    def set_error(self, error):
        self.error = error
        if self.verbatim == True :
            if error == Error.no_square_points:
                print('Could not find square in image')
            elif error == Error.no_inverse_matrix:
                print('Cannot create inverse matrix. Singular warping matrix. Probably bad tag detected!')

        return self.error

    def findWarpMatrix(self, model_tag): # returns 0 on succesfull matching

        if self.findSquare(model_tag) != Error.flawless:
            return self.error

        drawCentroid(self.scene_markuped, self.cntExternal, self.color_centroid) # DRAW centroid

        # self.mWarp2tag, mask= cv2.findHomography(src_pts, self.dst_pts, cv2.RANSAC, 5.0)

        # method = cv2.LMEDS
        src_pts = model_tag.ptsDetectArea
        self.mWarp2scene, _ = cv2.findHomography(src_pts, self.dst_pts, )

        # get inverse transformation matrix
        try:
            self.mWarp2tag = np.linalg.inv(self.mWarp2scene)
        except:
            # raise Exception('Cannot calculate inverse matrix.')

            # print("Cannot create inverse matrix. Singular warping matrix. Probably bad tag detected!")
            return self.set_error(Error.no_inverse_matrix)


        self.imWarped = self.drawSceneWarpedToTag(model_tag)

        self.addWarpRotation(model_tag)

        return self.error

    def addWarpRotation(self, model_tag):

        # find out if it is really a tag
        if model_tag.checkType == 'symbolSquareMeanValue':

            imSymbolArea = model_tag.symbolArea.getRoi( self.imWarped )

            imSymbolSubAreas = []
            for area in model_tag.symbolSubAreas:
                imSub = area.getRoi(imSymbolArea)
                imSymbolSubAreas.append(imSub)

            squareMeans = model_tag.getSquareMeans(imSymbolSubAreas)
            # print squareMeans
            # a = [1, 2, 3, 4]
            # b = [5,6,7,8]
            # print zip(a,b)
            # print(squareSums)
            # print len(cTagModel.imSymbolSubAreas)
            # print zip(imSymbolSubAreas, cTagModel.imSymbolSubAreas)
            # waitKeyExit()
            # print(squareMeans)

            self.rotation  = []
            for modelCode in model_tag.rotatedModelCodes:
                if modelCode == squareMeans: # * 1
                    self.rotation .append(1)
                else:
                    self.rotation .append(0)

            # print(self.rotation)
            if sum(self.rotation ) == 0:
                return self.set_error(Error.no_tag_rotations_found)
            if sum(self.rotation ) > 1:
                return self.set_error(Error.rotation_uncertainity)

            self.rotIdx = np.sum([ i*self.rotation[i] for i in range(0,4) ])
            self.mWarp2tag = matDot(model_tag.mInvRotTra[self.rotIdx], self.mWarp2tag)
        return self.error
        # thresholded element-wise addition
        # procentual histogram - of seenTag vs of tagModel

    def findSquare(self, model_tag):  # returns 0 on succesfull findings
        #
        # if self.cntExternal is None:
        #     # print("Should I count the cntExternal now?")
        #     self.calcExternalContour()
        #     if self.calcMoments() != 0:
                # print("Should I count the cntExternal now?")
                # return self.set_error(Error.external_contour_error)


        # self.calcExternalContour()

        def findFromCenter(cnt,im,mc):
            # rotated boundingbox
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            # return sq.findClosestToMinAreaRect(im,mc,box,cnt)
            # return sq.findFarthestFromCenter(im,mc,box,cnt)
            return sq.findClosestToMinAreaRectAndFarthestFromCenter(im, self.mc, box, cnt)
            # return box


        im = self.imScene
        cnt = self.cntExternal
        if len(cnt) < self.minimum_contour_length:
            # print('aa',cnt)
            return self.set_error(Error.contour_too_small)

        # tims = []
        # tims.append(Timeas())
        # corner_pts = findFromCenter(cnt, im, self.mc)
        # tims[-1].stop()
        #
        # tims.append(Timeas())
        # corner_pts = findDirectionDrift(cnt, self.external_contour_approx)
        # tims[-1].stop()
        #
        # tims.append(Timeas())

        # plt.ion()
        def make_gauss(im, a=5):
            return cv2.GaussianBlur(im, (a, a), 0)

        # cnt = self.getExternalContour( make_gauss(self.imScene))
        # cnt = self.getExternalContour( self.imScene)
        corner_pts = self.findApproxPolyDP(cnt)

        # corner_pts = findStableLineIntersection(cnt, self.external_contour_approx, plot= True, half_interval=1)
        # corner_pts = findStableLineIntersection(cnt, self.external_contour_approx, plot= False, half_interval=1)

        # time.sleep(10)

        # corner_pts = self.findMinAreaRectRecoursive(model_tag)
        # corner_pts = self.findMinAreaRect_StableLineIntersection(model_tag)

        # corners from FAST and then findStableLineIntersection
        # FAST conrenrs

        # contours inter and outer energies - only lines

        # houghlines

        # tims[-1].stop()

        # print('times','| '.join([tim.last() for tim in tims]))


        if corner_pts is None or len(corner_pts) != 4:
            return self.set_error(Error.no_square_points)

        # for corner_pt in corner_pts:
        #     for z in corner_pt:
        #         if z is float('nan'):
        #             self.set_error(Error.no_square_points)

        [self.set_error(Error.square_points_float_nan) for corner_pt in corner_pts for z in corner_pt if np.isnan(z)]
        if self.error != Error.flawless:
            return self.error


        self.dst_pts = np.array(corner_pts)
        drawDots(self.scene_markuped, self.dst_pts, self.color_corners) # draw corner points
        return self.error

    def findApproxPolyDP(self,cnt):

        # squares = []
        def angle_cos(p0, p1, p2):
            d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
            return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 10 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            # if max_cos < 0.1:
            return cnt

    def findMinAreaRect_StableLineIntersection(self, model_tag):

        cnt = self.getExternalContour(self.imScene)

        rect = cv2.minAreaRect(cnt)
        dst_pts = cv2.boxPoints(rect)
        src_pts = np.array(model_tag.ptsDetectArea)
        # print(type(src_pts))

        def get_it(src_pts, dst_pts):
            mWarp2scene, _ = cv2.findHomography(src_pts, dst_pts, )

            # get inverse transformation matrix
            try:
                mWarp2tag = np.linalg.inv(mWarp2scene)
            except:
                print("Cannot create inverse matrix. Singular warping matrix. MinAreaRect ")
                self.set_error(Error.no_inverse_matrix)
                return None

            imWarped = cv2.warpPerspective(self.imScene.copy(), mWarp2tag, model_tag.imTagDetect.shape,
                                           # flags = cv2.INTER_NEAREST )
                                               flags=cv2.INTER_LINEAR )
            return [mWarp2scene, mWarp2tag, imWarped]

        back = get_it(src_pts, dst_pts)
        if back is None:
            return None
        [minArea_to_scene, scene_to_minArea, imWarped1] = back


        def make_gauss(im, a=55):
            return cv2.GaussianBlur(im, (a, a), 0)

        imWarped1 = make_gauss(imWarped1)

        # cnt = self.getExternalContour(imWarped1)
        self.external_contour_approx = cv2.CHAIN_APPROX_TC89_L1
        # self.external_contour_approx = cv2.CHAIN_APPROX_TC89_KCOS
        cnt = self.getExternalContour(imWarped1)

        plot = False

        corners_in_warped = sq.findStableLineIntersection(cnt, self.external_contour_approx, plot=plot, half_interval=1)
        if corners_in_warped is None:
            return None
        print(corners_in_warped)

        back = get_it(np.array(corners_in_warped), dst_pts)
        if back is None:
            return None

        [right_to_minArea, minArea_to_right, imWarped2] = back

        # print(right_to_minArea.shape, minArea_to_scene.shape)
        a = right_to_minArea
        b = minArea_to_scene
        c = scene_to_minArea
        d = minArea_to_right
        x1 = a
        x2 = b
        xx = [a,b]
        # xx = [a,b,c]
        # xx = [a,b,d] #
        xx = [b,a,c] #
        xx = [b,a,d]
        xx = [a,c,b]
        xx = [a,c,d]
        xx = [a,d,b]
        xx = [a,d,c]
        xx = [b,a,c] #
        xx = [b,a,d]
        xx = [b,c,a]
        xx = [b,c,d]
        xx = [d,a,b]
        xx = [d,a,c]
        xx = [d,b,a]
        xx = [d,b,c]
        xx = [d,c,a]
        xx = [d,c,b]

        transformation = np.array(np.eye(3,3))
        for x in xx:
            transformation = matDot(transformation, np.array(x))
        # transformation = np.matrix(matDot(np.array(x1),
        #                                   np.array(x2)))

        transformation = np.matrix(transformation)

        if plot == True:
            plt.figure(1)
            sp = 311
            # imWarpeds = [[imWarped1], [imWarped2]]
            imWarpeds = [[self.imScene], [imWarped1]]
            for imWarped in imWarpeds:
                markuped_image = imWarped[0].copy()
                drawDots(markuped_image, dst_pts)
                plt.subplot(sp)
                sp += 1
                plt.imshow(markuped_image, cmap='gray')

            contoured_image = imWarped1.copy()
            drawContour(contoured_image , [cnt])
            plt.imshow(contoured_image) # , cmap='gray')
            plt.show()

        corners = []
        for q in range(4):
            mat_point = np.transpose(np.matrix([corners_in_warped[q][0], corners_in_warped[q][1], 0]))

            C = np.matrix(np.eye(3,1))
            np.dot(np.matrix(transformation), mat_point, C)
            xy = [C[0].tolist()[0], C[1].tolist()[0]]

            corners.append(xy)

        # print(corners)
        return corners


    def findMinAreaRectRecoursive(self, model_tag):

        src_pts = model_tag.ptsDetectArea
        # list of transformation matrices of individual recoursive rounds
        to_scene = []
        to_tag = []

        rounds = 10
        markuped_images = []

        tim = Timeas()
        image = self.imScene
        for q in range(rounds):
            cnt = self.getExternalContour(image)

            rect = cv2.minAreaRect(cnt)
            dst_pts = cv2.boxPoints(rect)

            mWarp2scene, _ = cv2.findHomography(src_pts, dst_pts, )

            to_scene.append(mWarp2scene)

            # get inverse transformation matrix
            try:
                mWarp2tag = np.linalg.inv(mWarp2scene)
            except:
                print("Cannot create inverse matrix. Singular warping matrix. In findMinAreaRectRecoursive, round:", q+1)
                self.set_error(Error.no_inverse_matrix)
                return None

            to_tag.append(mWarp2tag)

            imWarped = cv2.warpPerspective(image.copy(), mWarp2tag, model_tag.imTagDetect.shape,
                                               flags=cv2.INTER_LINEAR )



            image = imWarped.copy()

            markuped_image = imWarped.copy()
            drawDots(markuped_image, dst_pts)
            markuped_images.append([markuped_image])

        tim.print()

        plt.figure(1)
        rows = round(np.sqrt(rounds))
        cols = np.ceil(rounds/rows)
        sp = [rows, cols, 0]
        for q in range(rounds):
            sp[2] += 1
            plt.subplot(*sp)
            plt.imshow(markuped_images[q][0], cmap='gray')

        plt.show()

        transformation = np.eye(3)

        for q in range(rounds-1, 0, -1):
            transformation = matDot(transformation , to_tag[q])

        # transformation = to_scene[0]

        # only for drawing
        # find individual points in original scene
        corners = []
        for q in range(4):
            # vec_point = src_pts[0]
            # print(vec_point)
            mat_point = np.transpose(np.matrix([src_pts[q][0], src_pts[q][1], 0]))

            C = np.matrix(np.eye(3,1))
            #
            # print(mat_point)
            # print(transformation)
            # print(C)

            np.dot(np.matrix(transformation), mat_point, C)
            xy = [C[0].tolist()[0], C[1].tolist()[0]]

            corners.append(xy)

        # print(corners)
        return corners

        # # print (dst_pts)
        # return dst_pts


    def calculate(self, model_tag):

        # what was the purpose of this
        # if self.addExternalContour(self.cntExternal) != 0:
        #     print('added_external contour')
        #     continue
        # self.addExternalContour(self.cntExternal)
        self.findWarpMatrix(model_tag)
        # if self.findWarpMatrix(model_tag) == Error.flawless:
            # self.tag_warped = self.drawSceneWarpedToTag(model_tag)

    def drawTagWarpedToScene(self, imTag, imScene):
        h,w = imTag.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, self.mWarp2tag)
        return cv2.polylines(imScene,[np.int32(dst)],True, 128,3, cv2.LINE_8)

    def drawSceneWarpedToTag(self, model_tag):
        # print self.mInverse
        return cv2.warpPerspective(self.imScene, self.mWarp2tag, model_tag.imTagDetect.shape,
                                   flags=cv2.INTER_LINEAR )
                                    #, , cv2.BORDER_CONSTANT)

class Timeas:

    def __init__(self, type='s2ms'):
        self.start()
        self.set_output_type(type)

    def set_output_type(self, type):
        self.type = type

    def start(self):
        self.time_start = time.time()

    def stop(self):
        self.time_end = time.time()
        self.time_last = self.time_end - self.time_start

    def now(self):
        self.stop()
        return self.last()

    def print_last(self):
        print(self.last())

    def print(self):
        self.stop()
        print(self.last())

    def last(self):
        if self.type == 's2ms':
            return '{00:.2f} ms'.format(round(self.time_last * 1000,2))




class C_tagModel: # tag model
    def __init__(self, strTag):
        # later have function to get this from actual image

        if strTag == '2L':
            hwWhole = 250
            bSymbolArea = 60
            bDetectArea = 40

            self.whole = C_area([hwWhole ]*2, [0]*2)
            b = bSymbolArea
            self.symbolArea = C_area([hwWhole - b*2]*2,[b]*2)
            b = bDetectArea
            self.detectArea = C_area([hwWhole - b*2]*2,[b]*2)

            self.checkType = 'symbolSquareMeanValue'
            self.imTag = readIm('full', strTag)
            self.imTagDetect = readIm('invnoborder', strTag)
            self.imSymbolArea = self.symbolArea.getRoi(self.imTagDetect)
            self.imDetectArea = self.detectArea.getRoi(self.imTagDetect)

            self.ptsSymbolArea = getBoxCorners(self.symbolArea.tl[0], self.symbolArea.hw[0] )
            self.ptsDetectArea = getBoxCorners(self.detectArea.tl[0], self.detectArea.hw[0] )

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # detection of square subAreas in SymbolArea
            num = 2

            self.symbolSubAreas= self.symbolArea.getSubAreas(num, num)
            self.rotatedModelCodes = []
            for rot in range(0,4):
                imSymbolSubAreas = []
                for area in self.symbolSubAreas:
                    imSub = area.getRoi( np.rot90(self.imSymbolArea, rot ) )
                    imSymbolSubAreas.append(imSub)

                modelCode = self.getSquareMeans(imSymbolSubAreas)
                self.rotatedModelCodes.append(modelCode)

            # 90 degree rotation preprocessed matrices
            [dx, dy] = [-d / 2 for d in self.imTag.shape]
            self.mTra = np.array([ [1,0,dx], [0,1,dy], [0,0,1] ])
            self.mTraInv = np.array([ [1,0,-dx], [0,1,-dy], [0,0,1] ])

            self.mRotTra = []
            self.mInvRotTra = []
            for angleIdx in range(0,4): # 0, 90, 180, 270 in counterclockwise?
                angle = np.deg2rad(angleIdx*90)

                cos = np.cos(angle)
                sin = np.sin(angle)
                mRot = np.array([ [cos,   sin,   0], [-sin,   cos,    0], [0,     0,      1] ])
                mRotTra = matDot(self.mTraInv , matDot(mRot, self.mTra) )
                try:
                    mInvRotTra = np.linalg.inv( mRotTra )
                except:
                    raise Exception('Cannot calculate inverse matrix.')
                    # print("Cannot create inverse matrix. Singular warping matrix. Probably bad tag detected!")
                    return 1
                self.mRotTra.append(mRotTra)
                self.mInvRotTra.append( mInvRotTra )




    def getSquareMeans(self, imSymbolSubAreas):
        # return [  np.float(np.sum(imSub))
        #           for imSub in imSymbolSubAreas ]
        max = np.float(np.max(imSymbolSubAreas))
        if max == 0:
            max = 1
        return [  np.int( np.round(
                np.sum(imSub) / (imSub.shape[0] * imSub.shape[1]) / max
                ) )
                for imSub in imSymbolSubAreas ]

        # for imSub in imSymbolSubAreas:
        #     print(np.sum(imSub))

    #
    # def __init__(self, area):
    #     # later have function to get this from actual image
    #
    #     if strTag = '2L':
    #         self.width = 250
    #         self.height = 250
    #         self.left = 0
    #         self.top = 0

    # if strTag != 2:
    #     return
    # # symbol square - most inner = A
    # # symbol square frame 1 - 2nd most inner = B


class C_area:
    def __init__(self, hw, tl):
        self.hw = hw
        self.tl = tl

    def getRoi(self,im):
        return im[  self.tl[0] : self.tl[0] + self.hw[0],
                    self.tl[1] : self.tl[1] + self.hw[1] ]

    def getSubAreas(self, rows, cols):
        # one cell dimensions
        hSub = int(self.hw[0] / rows)
        wSub = int(self.hw[1] / cols)

        # border pixels vertical
        hSubMulti = hSub * rows
        if hSubMulti < self.hw[0]:
            # must append - > append to the last one
            hDiff = self.hw[0] - hSubMulti
        else:
            hDiff = 0

        # border pixels horizontal
        wSubMulti = wSub * cols
        if wSubMulti < self.hw[1]:
            # must append - > append to the last one
            wDiff = self.hw[1] - wSubMulti
        else:
            wDiff = 0

        # create the subareas
        aSubs = []
        hw = (hSub, wSub)
        for iRow in range(0,rows):
            for iCol in range(0,cols):
                tl = (iRow*hSub, iCol*wSub)
                aSub = C_area( hw, tl)
                if iRow == rows-1:
                    if iCol == cols-1:
                        aSub = C_area( (hSub+hDiff,wSub+wDiff), tl)
                    else:
                        aSub = C_area( (hSub,wSub+wDiff), tl)
                if iCol == cols-1:
                    aSub = C_area( (hSub,wSub+wDiff), tl)
                # print aSub.hw
                # print aSub.tl
                aSubs.append(aSub)

        return aSubs
    # def __init__(self, hw, tl, tlFromUpperHW = False):
    #     self.hw = hw
    #     if tlFromUpperHW == False:
    #         self.tl = tl
    #     else:
    #         self.tl = self.getTopLeftCentered(tl)
#
#     def __init__(self, hw, hwWhole):
#         self.height = h
#         self.width = w
#         [self.top, self.left] = getTopLeftCentered()
#
#     def getTopLeftCentered(self,hwUpper):
#         return self.




def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def readIm(pre, tag):
    dIm = './pic/'
    fIm =  pre + '_' + tag + '.png'
    im = cv2.imread(dIm + fIm,0)
    if im is not None:
        print("Loaded image: [" + fIm + "] = " + str(im.shape) )
    return im

def getBoxCorners(boxOffset, boxSide):
    aS = boxOffset
    aB = boxOffset + boxSide
    pts = [[aS, aS], [aS, aB], [aB, aB], [aB, aS]]
    return np.float32(pts)

def read_model_tag(strTag):
    cTag = C_tagModel(strTag)
    return cTag

def makeBorder(im, bgColor):
    bs = max(im.shape)
    im = cv2.copyMakeBorder(im, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=bgColor)
    return im, bs

def makeLaplacian(im):
    return np.uint8(np.absolute(cv2.Laplacian(im, cv2.CV_64F)))

def joinTwoIm(imBig,imSmall, vertically = 0, color = 0):
    diff = imBig.shape[vertically] - imSmall.shape[vertically]
    if vertically == 0:
        imEnlarged = cv2.copyMakeBorder(imSmall, 0,diff, 0,   0, cv2.BORDER_CONSTANT, value=color)
        return np.hstack([imBig,imEnlarged])
    else:
        imEnlarged = cv2.copyMakeBorder(imSmall, 0,   0, 0,diff, cv2.BORDER_CONSTANT, value=color)
        return np.vstack([imBig,imEnlarged])

def joinIm(ims, vertically = 0, color = 0):
    imLast = []
    for im in ims:
        im = im[0]
        if imLast != []:
            # print(vertically)
            if (im.shape[vertically] - imLast.shape[vertically]) > 0:
               # im is bigger
               imLast = joinTwoIm(im,imLast,vertically, color)
            else:
               imLast = joinTwoIm(imLast,im,vertically, color)
        else:
            imLast = im

    return np.array(imLast)

def colorifyGray(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

def colorify(im):
    if len(im.shape) == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    else:
        return im.copy()

def drawContour(im, cnt, color = 180, thickness = 1):
    cv2.drawContours(im, cnt, 0, color, thickness)

def drawDots(im, dots, numbers=1):
    i = 0
    for dot in dots:
        pt = [int(dot[0]), int(dot[1])]
        # col = (255, 0, 0)
        col = 180
        sh_max = np.max(im.shape)
        # radius = np.int(sh_max  / 40)
        radius = 1
        thickness = np.int(sh_max  / 140)
        cv2.circle(im, tuple(pt), radius, col, thickness )
        numbers = 1
        if numbers == 1:
            # font = cv2.FONT_HERSHEY_SIMPLEX
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            cv2.putText(im,str(i), tuple([ d+10 for d in pt ]), font, 1, 0, thickness+2 )
            cv2.putText(im,str(i), tuple([ d+10 for d in pt ]), font, 1, 255, thickness )
        i += 1
    return im

def drawBoundingBox(im,cnt, color = 255, lineWidth = 1):
    # non-rotated boundingbox
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(im, (x, y), (x + w, y + h), color, lineWidth)

def drawRotatedBoundingBox(im, cnt, color = 255, lineWidth = 1):
    # draw rotated minAreaRect boundingBox
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    int_box = np.int0(box)
    cv2.drawContours(im,[int_box],0,color,1)

def drawCentroid(im, cnt, color = 255):
    mu = cv2.moments(cnt)
    mc = getCentralMoment(mu)
    cv2.circle(im, tuple(int(i) for i in mc), 4, color, -1, 8, 0)

def getCentralMoment(mu):
    if mu['m00'] == 0:
        raise Exception('Moment of image m00 is zero. Could not count central moment!')
    return [ mu['m10'] / mu['m00'],
             mu['m01'] / mu['m00'] ]


def matDot(A,B):
    C = np.eye(3,3)
    np.dot(A,B,C)
    return C


if __name__ == '__main__':

    # cv2.destroyAllWindows()

    pass



def waitKeyExit():
    while True:
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        if k == 27:
            break
