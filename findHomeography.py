import numpy as np
import cv2

class C_observedTag:
    # static class variable - known Tag
    # tagModels = loadTagModels('2L')

    def __init__(self, imTagInScene):
        self.imScene = imTagInScene # image of scene in which the tag is supposed to be
        self.imWarped = None # ground floor image of tag transformed from imScene
        self.dst_pts = None # perspectively deformed detectionArea square corner points
        self.mWarp2scene = None # transformation matrix from scene
        self.mInverse = None
        self.cntExternal = None # detectionArea external contour
        self.mu = None
        self.mc = None
        self.rotation = None
        # if cntExternal is not None:
        #     cntExternal

        # else:

    def calcMoments(self):  # returns 0 on success
        self.mu = cv2.moments(self.cntExternal)
        if self.mu['m00'] == 0:
            return 1
        self.mc = np.float32(getCentralMoment(self.mu))
        return 0

    def calcExternalContour(self): # returns 0 on success
        _, contours, hierarchy = cv2.findContours(self.imScene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.cntExternal = contours[0]
        return self.calcMoments()

    def addExternalContour(self,cntExternal): # returns 0 on success
        self.cntExternal = cntExternal
        return self.calcMoments()

    def findWarpMatrix(self, cTagModel): # returns 0 on succesfull matching
        src_pts = cTagModel.ptsDetectArea
        if self.findSquare() != 0:
            # print('Could not find square in image')
            return 1
        # self.mWarp2scene, mask= cv2.findHomography(src_pts, self.dst_pts, cv2.RANSAC, 5.0)
        self.mWarp2scene, _ = cv2.findHomography(src_pts, self.dst_pts, cv2.LMEDS)
        # matchesMask = mask.ravel().tolist()
        # what is mask ?!

        # get inverse transformation matrix
        try:
            self.mInverse = np.linalg.inv(self.mWarp2scene)
        except:
            # raise Exception('Cannot calculate inverse matrix.')
            # print("Cannot create inverse matrix. Singular warping matrix. Probably bad tag detected!")
            return 1

        self.imWarped = self.drawSceneWarpedToTag(cTagModel)

        check = self.addWarpRotation(cTagModel)
        if check != 0:
            return 1


        return 0

    def addWarpRotation(self,cTagModel):

        # find out if it is really a tag
        if cTagModel.checkType == 'symbolSquareMeanValue':

            imSymbolArea = cTagModel.symbolArea.getRoi( self.imWarped )

            imSymbolSubAreas = []
            for area in cTagModel.symbolSubAreas:
                imSub = area.getRoi(imSymbolArea)
                imSymbolSubAreas.append(imSub)

            squareMeans = cTagModel.getSquareMeans(imSymbolSubAreas)
            # print squareMeans

            self.rotation  = []
            for modelCode in cTagModel.rotatedModelCodes:
                if modelCode == squareMeans:
                    self.rotation .append(1)
                else:
                    self.rotation .append(0)

            # print rotation
            if sum(self.rotation ) == 0:
                return 1
            if sum(self.rotation ) > 1:
                return 2 # two or more possible rotations

            self.rotIdx = np.sum([ i*self.rotation[i] for i in range(0,4) ])
            self.mInverse = matDot(cTagModel.mInvRotTra[self.rotIdx], self.mInverse)
        return 0
        # thresholded element-wise addition
        # procentual histogram - of seenTag vs of tagModel

    def findSquare(self):  # returns 0 on succesfull findings
        if self.cntExternal is None:
            # print("Should I count the cntExternal now?")
            if self.calcExternalContour() != 0:
                return 1

        im = self.imScene
        cnt = self.cntExternal

        # rotated boundingbox
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        # corner_pts = findClosestToMinAreaRect(im,mc,box,cnt)
        # corner_pts = findFarthestFromCenter(im,mc,box,cnt)
        corner_pts = findClosestToMinAreaRectAndFarthestFromCenter(im,self.mc,box,cnt)

        if corner_pts == []:
            return 1

        self.dst_pts = corner_pts
        return 0


    def drawTagWarpedToScene(self, imTag, imScene):
        h,w = imTag.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, self.mWarp2scene)
        return cv2.polylines(imScene,[np.int32(dst)],True, 128,3, cv2.LINE_8)

    def drawSceneWarpedToTag(self, cTagModel):
        # print self.mInverse
        return cv2.warpPerspective(self.imScene.copy(), self.mInverse, cTagModel.imTagDetect.shape,
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
                    # raise Exception('Cannot calculate inverse matrix.')
                    # print("Cannot create inverse matrix. Singular warping matrix. Probably bad tag detected!")
                    return 1
                self.mRotTra.append(mRotTra)
                self.mInvRotTra.append( mInvRotTra )




    def getSquareMeans(self, imSymbolSubAreas):
        return [  np.int( np.round(
                np.sum(imSub) / (imSub.shape[0]*imSub.shape[1]) / 255.0
                ) )
                for imSub in imSymbolSubAreas ]

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

def readTag(strTag):
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
            if (im.shape[vertically] - imLast.shape[vertically]) > 0:
               # im is bigger
               imLast = joinTwoIm(im,imLast,vertically, color)
            else:
               imLast = joinTwoIm(imLast,im,vertically, color)
        else:
            imLast = im

    return imLast

def colorify(im):
    return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

def drawDots(im, dots, numbers=1):
    i = 0
    for dot in dots:
        pt = [int(dot[0]),int(dot[1])]
        # col = (255, 0, 0)
        col = 180
        cv2.circle(im, tuple(pt), 4, col, 1)
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

def findClosestToMinAreaRectAndFarthestFromCenter(im,mc,box,cnt):
   # find points from countour which are the closest (L2SQR) to minAreaRect & also farthest from center
    norm = cv2.NORM_L2SQR
    mc = np.float32(mc)
    corner_pts = []
    [corner_pts.append(box[i]) for i in range(0,4)] # append 4 mc
    corner_pts = np.float32(corner_pts)

    distSq = [0] *4 # distance = distFromCenter - distFromMinBox
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

def findExtremes(cnt):
    extremes = []
    extremes.extend([cnt[cnt[:,:,0].argmin()][0]] ) # leftmost
    extremes.extend([cnt[cnt[:,:,1].argmin()][0]] ) # topmost
    extremes.extend([cnt[cnt[:,:,0].argmax()][0]] ) # rightmost
    extremes.extend([cnt[cnt[:,:,1].argmax()][0]] ) # bottommost
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

    strTag = '2L'
    cTagModel = readTag(strTag)

    imScene = readIm('space1', strTag)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cSeenTag = C_observedTag(imScene.copy())
    success = cSeenTag.findWarpMatrix(cTagModel)
    if not success:
        print 'Tag from scene could not be transformed!'
        exit

    imTagRecreated = cSeenTag.drawSceneWarpedToTag(cTagModel)

    mWarp = cSeenTag.mWarp2scene

    print(cTagModel.ptsDetectArea)
    print(cSeenTag.dst_pts)
    imScene = drawDots(imScene, cSeenTag.dst_pts)
    # imScene = drawRotatedBoundingBox(imScene,cnt,)

    print(str(mWarp) + " = Homography transformation matrix")

    imTag = drawDots(cTagModel.imTagDetect.copy(), cTagModel.ptsDetectArea)
    imTagFromScene = cSeenTag.drawSceneWarpedToTag(cTagModel)

    imBoth = joinIm([ [imTagFromScene], [imTag], [imScene] ])

    imAll = colorify(imBoth)

    # a = 0.5
    # imAll = cv2.resize(imAll, (0, 0), fx=a, fy=a)
    cv2.imshow('images',imAll)

    a = 0
    while 1:
        a = a + 1
        if a > 200:
            break
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        if k == 27:
            break
    cv2.destroyAllWindows()





def waitKeyExit():
    while True:
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        if k == 27:
            break