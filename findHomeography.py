import numpy as np
import cv2
from enum import Enum

from itertools import cycle


class Error(Enum):
    flawless = 0
    no_square_points = 1
    no_inverse_matrix = 2
    no_tag_rotations_found = 3
    rotation_uncertainity = 4
    external_contour_missunderstanding_probably = 5
    contour_too_small = 6
    some1 = 7


class C_observedTag:
    # static class variable - known Tag
    # tagModels = loadTagModels('2L')

    def __init__(self, imTagInScene, scene_markuped):
        self.imScene = imTagInScene # image of scene in which the tag is supposed to be
        self.imWarped = None # ground floor image of tag transformed from imScene
        # self.tag_warped = None # warped tag image into model_tag space

        self.dst_pts = None # perspectively deformed detectionArea square corner points
        self.mWarp2tag = None # transformation matrix from perspective scene to ground floor tag
        self.mWarp2scene = None # transformation matrix from ground floor tag to perspective scene
        self.cntExternal = None # detectionArea external contour
        self.mu = None # all image moments
        self.mc = None # central moment
        self.rotation = None # square symbols in symbolArea check possible rotations similar to cTagModel - [0,90,180,270]deg
        self.error = Error.flawless # error
        self.scene_markuped = scene_markuped # whole image to print additive markups of this observed tag to

        self.color_corners = 160
        self.color_centroid = 180
        self.external_contour_approx = cv2.CHAIN_APPROX_SIMPLE
        # self.external_contour_approx = cv2.CHAIN_APPROX_TC89_L1

        self.minimum_contour_length = 4*4

    def calcMoments(self):  # returns 0 on success
        self.mu = cv2.moments(self.cntExternal)
        if self.mu['m00'] == 0:
            return 1
        self.mc = np.float32(getCentralMoment(self.mu))
        return 0

    def calcExternalContour(self): # returns 0 on success

        _, contours, hierarchy = cv2.findContours(self.imScene.copy(), cv2.RETR_EXTERNAL, self.external_contour_approx )
        self.cntExternal = contours[0]
        return self.calcMoments()

    def addExternalContour(self, cntExternal): # returns 0 on success
        self.cntExternal = cntExternal
        return self.calcMoments()

    def set_error(self, error):
        self.error = error
        verbatim = True
        if verbatim == True :
            if error == Error.no_square_points:
                print('Could not find square in image')
            elif error == Error.no_inverse_matrix:
                print('Cannot create inverse matrix. Singular warping matrix. Probably bad tag detected!')

        return self.error

    def findWarpMatrix(self, model_tag): # returns 0 on succesfull matching

        if self.findSquare() != Error.flawless:
            return self.error

        drawCentroid(self.scene_markuped, self.cntExternal, self.color_centroid) # DRAW centroid

        # self.mWarp2tag, mask= cv2.findHomography(src_pts, self.dst_pts, cv2.RANSAC, 5.0)
        # method = cv2.LMEDS
        src_pts = model_tag.ptsDetectArea

        self.mWarp2scene, _ = cv2.findHomography(src_pts, self.dst_pts, )
        # matchesMask = mask.ravel().tolist()
        # what is mask ?!

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

    def findSquare(self):  # returns 0 on succesfull findings
        if self.cntExternal is None:
            # print("Should I count the cntExternal now?")
            if self.calcExternalContour() != 0:
                print("Should I count the cntExternal now?")
                self.set_error(Error.external_contour_missunderstanding_probably)
                return 42



        def findFromCenter(cnt,im,mc):
            # rotated boundingbox
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            # return findClosestToMinAreaRect(im,mc,box,cnt)
            # return findFarthestFromCenter(im,mc,box,cnt)
            return findClosestToMinAreaRectAndFarthestFromCenter(im, self.mc, box, cnt)

        im = self.imScene
        cnt = self.cntExternal
        # corner_pts = findFromCenter(cnt, im, self.mc)

        if len(cnt) < self.minimum_contour_length:
            return self.set_error(Error.contour_too_small)

        corner_pts = findDirectionDrift(cnt, self.external_contour_approx)

        if corner_pts == []:
            return self.set_error(Error.no_square_points)

        self.dst_pts = corner_pts
        drawDots(self.scene_markuped, self.dst_pts, self.color_corners) # draw corner points
        return self.error

    def calculate(self, model_tag):

        # what was the purpose of this
        # if self.addExternalContour(self.cntExternal) != 0:
        #     print('added_external contour')
        #     continue
        self.addExternalContour(self.cntExternal)
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

def drawDots(im, dots, numbers=1):
    i = 0
    for dot in dots:
        pt = [int(dot[0]),int(dot[1])]
        # col = (255, 0, 0)
        col = 180
        radius = 10
        cv2.circle(im, tuple(pt), radius, col, 1)
        if numbers == 1:
            # font = cv2.FONT_HERSHEY_SIMPLEX
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            cv2.putText(im,str(i), tuple([ d+10 for d in pt ]), font, 1, 0, 3 )
            cv2.putText(im,str(i), tuple([ d+10 for d in pt ]), font, 1, 255, 1 )
        i += 1
    return im

def findClosestToMinAreaRect(im,mc,box,cnt):
    # find points from countour which are the closest (L2SQR) to minAreaRect!
    norm = cv2.NORM_L2SQR
    mc = np.float32(mc)
    corner_pts = []
    [corner_pts.append(box[i]) for i in range(0,4)] # append 4 mc
    #corner.append [mc, dist]
    corner_pts = np.float32(corner_pts)
    # print(corner_pts)

    distSq = [] # distance between corner_pts and minAreaRect pts
    [distSq.append(cv2.norm(mc, corner_pt, norm)) for corner_pt in corner_pts] # initialize to distance to center (mc)
    distSq = np.float32(distSq)
    # print(distSq)

    cnt = np.float32(cnt)
    # print('starting to count')

    for pt in cnt:
        cnt_pt = pt[0]
        for i in range(0,4):

            dist = cv2.norm(cnt_pt, box[i], norm)
            if dist < distSq[i]:
                distSq[i] = dist
                corner_pts[i] = cnt_pt

                # print('cnt_pt =' + str(cnt_pt))
                # print('corner_pts['+str(i)+'] = ' + str(corner_pts[i]))
                # print('dist = ' + str(dist))
                # print('distSq[i] = ' + str(distSq[i]))
                # print('took new cnt_pt which is closer '+ str(dist) + ' than the previous ' +str(distSq[i]))
                # print('____________________________________________________')
    # draw minAreaRect closest rectangle
    color = 150
    int_box = np.int0(corner_pts)
    cv2.drawContours(im,[int_box],0,color,1)
    return corner_pts

def findFarthestFromCenter(im,mc,box,cnt):
    # find points from countour which are the closest (L2SQR) to minAreaRect!
    norm = cv2.NORM_L2SQR
    mc = np.float32(mc)
    corner_pts = []
    [corner_pts.append(mc) for i in range(0,4)] # append 4 mc
    #corner.append [mc, dist]
    corner_pts = np.float32(corner_pts)
    # print(corner_pts)

    distSq = [0] *4 # distance between corner_pts and mc
    distSq = np.float32(distSq)
    # print(distSq)

    cnt = np.float32(cnt)
    # print('starting to count')

    for pt in cnt:
        cnt_pt = pt[0]
        for i in range(0,4):

            dist = cv2.norm(cnt_pt, mc, norm)
            if dist > distSq[i]:
                distSq[i] = dist
                corner_pts[i] = cnt_pt

                # print('cnt_pt =' + str(cnt_pt))
                # print('corner_pts['+str(i)+'] = ' + str(corner_pts[i]))
                # print('dist = ' + str(dist))
                # print('distSq[i] = ' + str(distSq[i]))
                # print('took new cnt_pt which is farther ' + str(dist) + ' than the previous ' +str(distSq[i]))
                # print('____________________________________________________')
    # draw minAreaRect closest rectangle
    color = 150
    int_box = np.int0(corner_pts)
    cv2.drawContours(im,[int_box],0,color,1)
    return corner_pts

def findClosestToMinAreaRectAndFarthestFromCenter(im, mc, box, cnt):
   # find points from countour which are the closest (L2SQR) to minAreaRect & also farthest from center
    norm = cv2.NORM_L2SQR
    mc = np.float32(mc)
    corner_pts = []
    [corner_pts.append(box[i]) for i in range(0,4)] # append 4 mc
    corner_pts = np.float32(corner_pts)

    distSq = [0,0,0,0] # distance = distFromCenter - distFromMinBox
    distSq = np.float32(distSq)

    cnt = np.float32(cnt)
    for pt in cnt:
        cnt_pt = pt[0]
        for i in range(0,4):
            distFromMinBox = cv2.norm(cnt_pt, box[i], norm)
            distFromCenter = cv2.norm(cnt_pt, mc, norm)
            dist = distFromCenter - distFromMinBox
            if dist > distSq[i]:
                distSq[i] = dist
                corner_pts[i] = cnt_pt

                # print('cnt_pt =' + str(cnt_pt))
                # print('corner_pts['+str(i)+'] = ' + str(corner_pts[i]))
                # print('dist = ' + str(dist))
                # print('distSq[i] = ' + str(distSq[i]))
                # print('took new cnt_pt which is closer '+ str(dist) + ' than the previous ' +str(distSq[i]))
                # print('____________________________________________________')
    # draw minAreaRect closest rectangle
    # color = 150
    # int_box = np.int0(corner_pts)
    # cv2.drawContours(im,[int_box],0,color,1)
    return corner_pts

# from pylab import *
import matplotlib.pyplot as plt
import matplotlib

# plt.ion()

def findDirectionDrift(cnt, external_contour_approx):
    """
    Find points from countour which have the biggest change in direction of three consecutive pixels
    """
    norm = cv2.NORM_L2SQR

    if external_contour_approx != cv2.CHAIN_APPROX_SIMPLE:
        print('Cannot find contour direction drift from not continuous external contour')
        return None

    half_interval = 1
    vy = []
    vx = []
    inv = []
    # circular list
    count = len(cnt)
    cnt = np.float32(cnt)
    direction = np.eye(1,1)

    plt.figure(1)
    plt.clf()
    sp = 510


    for q in range(count):
        first = q - half_interval
        last = q + half_interval
        contour_segment = np.array( [cnt[k % count][0] for k in range(first, last + 1)] )
        [_vx, _vy, _, _] = cv2.fitLine(contour_segment, norm, 0, 0.5, 0.5)

        vec_ac = cnt[first % count][0] - cnt[last % count][0]
        # print(vec_ac)
        coef = 1
        if vec_ac[1] < 0:
            coef = -1
            # print('-1')
        # vec1 = np.matrix([cnt[first % count][0] - cnt[q][0]])
        # vec2 = np.matrix([cnt[q][0] - cnt[last % count][0]])
        #
        # np.cross(vec1, vec2, direction)
        #
        # if np.sum(vec1 - vec2) != 0:
        #     print('1', vec1, '2', vec2, 'd', direction)
        #
        # if direction < 0:
        #     _vx *= -1
        #     print('not convex')


        # if contour_segment
        vy.append(_vy)
        vx.append(_vx)
        inv.append(coef)


    angles = np.arctan2(np.array(vy), np.array(vx)).tolist()

    sp += 1
    plt.subplot(sp)
    plt.plot(inv)
    plt.ylabel('inv ')

    sp += 1
    plt.subplot(sp)
    plt.plot(angles)
    plt.ylabel('angles atan2')

    # # normalize
    # pihalf = np.pi / 2
    # angles = [angle[0] + pihalf for angle in angles]
    angles = [angle[0] for angle in angles]

    for q in range(count):
        if inv[q] > 0:
            angles[q] += np.pi
            # print('less')


    print(count)

    sp += 1
    plt.subplot(sp)
    plt.plot(angles)
    plt.ylabel('angles')

    # we want raising positive angles - so derivation will be positive -> so we want to find minimum
    # todo check whether the angles are raising
    # not exactly discrete derivation
    derivate = [ angles[(q + 1) % count] - angles[(q) % count] for q in range(count)]
    # when subtracting 2pi -> 0.1 its negative -> we want it to be positive "and small"

    sp += 1
    plt.subplot(sp)
    plt.plot(derivate)
    plt.ylabel('derivates not normalized')
    #
    # npi2 = 2 * np.pi
    #
    # for q in range(count):
    #     if derivate[q] < 0:
    #         derivate[q] += npi2

    # derivate = [ derivate[(q + 1) % count] - derivate[(q - 1) % count] for q in range(count)]
    #
    # sp += 1
    # plt.subplot(sp)
    # plt.plot(derivate)
    # plt.ylabel('derivates')

    # corner_indexes = np.argpartition(np.array(derivate), -4)
    # corner_indexes = np.partition(np.array(derivate), 4)[:4]
    diff = np.array(derivate)
    sorted_indexes = np.argpartition(diff,-3)
    max_indexes = (sorted_indexes[-3:]).tolist()
    min_index = [np.argmin(diff)]
    # print(sorted_indexes)


    sorted = diff[sorted_indexes]
    sp += 1
    plt.subplot(sp)
    plt.plot(sorted)
    plt.ylabel('sorted')

    # find one minimum and 3 maximums
    corner_indexes = min_index + max_indexes

    print(corner_indexes)
    # #
    # plt.show()
    # plt.draw()


    if len(corner_indexes) > 0:
        return np.array([cnt[k][0] for k in corner_indexes])
    else:
        return None


def findExtremes(cnt):
    extremes = []
    extremes.extend([cnt[cnt[:, :, 0].argmin()][0]] ) # leftmost
    extremes.extend([cnt[cnt[:, :, 1].argmin()][0]] ) # topmost
    extremes.extend([cnt[cnt[:, :, 0].argmax()][0]] ) # rightmost
    extremes.extend([cnt[cnt[:, :, 1].argmax()][0]] ) # bottommost
    return extremes

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

    # strTag = '2L'
    # cTagModel = read_model_tag(strTag)
    #
    # imScene = readIm('space1', strTag)
    #
    # # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # cSeenTag = C_observedTag(imScene.copy())
    # success = cSeenTag.findWarpMatrix(cTagModel)
    # if not success:
    #     print('Tag from scene could not be transformed!')
    #     exit
    #
    # imTagRecreated = cSeenTag.drawSceneWarpedToTag(cTagModel)
    #
    # mWarp = cSeenTag.mWarp2tag
    #
    # print(cTagModel.ptsDetectArea)
    # print(cSeenTag.dst_pts)
    # imScene = drawDots(imScene, cSeenTag.dst_pts)
    # # imScene = drawRotatedBoundingBox(imScene,cnt,)
    #
    # print(str(mWarp) + " = Homography transformation matrix")
    #
    # imTag = drawDots(cTagModel.imTagDetect.copy(), cTagModel.ptsDetectArea)
    # imTagFromScene = cSeenTag.drawSceneWarpedToTag(cTagModel)
    #
    # imBoth = joinIm([ [imTagFromScene], [imTag], [imScene] ])
    #
    # imAll = colorifyGray(imBoth)
    #
    # # a = 0.5
    # # imAll = cv2.resize(imAll, (0, 0), fx=a, fy=a)
    # cv2.imshow('images',imAll)
    #
    # a = 0
    # while 1:
    #     a = a + 1
    #     if a > 200:
    #         break
    #     k = cv2.waitKey(30) & 0xff
    #     if k == ord('q'):
    #         break
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()

    pass



def waitKeyExit():
    while True:
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        if k == 27:
            break
