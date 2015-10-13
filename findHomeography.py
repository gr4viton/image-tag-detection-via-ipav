import numpy as np
import cv2

class C_observedTag:
    # static class variable - known Tag
    # tagModels = loadTagModels('2L')

    def __init__(self, imTagInScene):
        self.imScene = imTagInScene
        self.imWarped = None
        self.dst_pts = None
        self.mWarp = None
        self.mInverse = None
        self.cntExternal = None
        self.mu = None
        self.mc = None
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
        # self.mWarp, mask= cv2.findHomography(src_pts, self.dst_pts, cv2.RANSAC, 5.0)
        self.mWarp, _ = cv2.findHomography(src_pts, self.dst_pts, cv2.LMEDS)
        # matchesMask = mask.ravel().tolist()
        # what is mask ?!

        # get inverse transformation matrix
        try:
            self.mInverse = np.linalg.inv(self.mWarp)
        except:
            # raise Exception('Cannot calculate inverse matrix.')
            print("Cannot create inverse matrix. Singular warping matrix. Probably bad tag detected!")
            return 1

        self.imWarped = self.drawSceneWarpedToTag(cTagModel)

        # find out if it is really a tag
        d = cTagModel.detectArea
        self.imWarped = self.drawSceneWarpedToTag(cTagModel)
        # detectArea = cTagModel.symbolArea.getRoi( self.imWarped )


        # print cTagModel.ptsSymbolArea
        imSymbolArea = cTagModel.symbolArea.getRoi( self.imWarped )
        # cv2.imshow('symbolArea',imSymbolArea )


        aSubs = cTagModel.symbolSubAreas
        # print aSubs

        imSymbolSubAreas = []
        for area in aSubs:
            imSub = area.getRoi(self.imWarped)
            imSymbolSubAreas.append(imSub)

        squareSums = \
            [ [ np.sum(imSub) / (imSub.shape[0]*imSub.shape[1]) ,
               np.sum(imSubModel) / (imSubModel.shape[0]*imSubModel.shape[1]) ]
            for imSub, imSubModel
            in zip(imSymbolSubAreas, cTagModel.imSymbolSubAreas)]

        # a = [1, 2, 3, 4]
        # b = [5,6,7,8]
        # print zip(a,b)
        print squareSums
        # print len(cTagModel.imSymbolSubAreas)
        # print zip(imSymbolSubAreas, cTagModel.imSymbolSubAreas)
        # waitKeyExit()

        # thresholded element-wise addition
        # procentual histogram - of seenTag vs of tagModel


        # mWarp = addWarpRotation(mWarp, cTagModel, imScene)
        return 0


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

        # print rect
        # print box

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
        dst = cv2.perspectiveTransform(pts, self.mWarp)
        return cv2.polylines(imScene,[np.int32(dst)],True, 128,3, cv2.LINE_8)

    def drawSceneWarpedToTag(self, cTagModel):
        # print self.mInverse
        return cv2.warpPerspective(self.imScene.copy(), self.mInverse, cTagModel.imTagDetect.shape) #, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    def addWarpRotation(self,imScene,imTag):

        # findTagRotation()
        # find affine rotation
        angle = np.deg2rad(90)
        cos = np.cos(angle)
        sin = np.sin(angle)
        # rotMatrix = np.array([[cos, -sin, 0], [sin, cos, 0], [0,0,1] ])
        mRot = np.array([
            [cos,   -sin,   0],
            [sin,   cos,    0],
            [0,     0,      1] ])
        # rotMatrix = np.eye(3,3)

        dx = -imTag.shape[0] / 2
        dy = -imTag.shape[1] / 2

        dx = -imScene.shape[0] / 2
        dy = -imScene.shape[1] / 2
        # dx = 10
        # dy = 10
        mTra = np.array([
            [1,0,dx],
            [0,1,dy],
            [0,0,1]
        ])
        mTraInv = np.linalg.inv(mTra)


        mFinal = self.mInverse.copy()
        # mFinal = np.eye(3,3)

        # np.dot(mFinal, mTra, mFinal )
        # np.dot(mFinal, mRot, mFinal)
        # np.dot(mFinal, mTraInv, mFinal )
        # np.dot(mInverse, mFinal, mFinal)

        # mFinal = matDot(mInverse,mFinal)

        # mFinal = matDot(mInverse, mFinal)
        # mFinal = matDot(mFinal, mTra)
        # # mFinal = matDot(mFinal, mRot)
        # mFinal = matDot(mFinal, mTraInv)
        #
        # print("mInverse")
        # print(mInverse)
        # print("mRot")
        # print(mRot)
        # print("mFinal")
        # print(mFinal)

        return mFinal


class C_tag: # tag model
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

            self.imTag = readIm('full', strTag)
            self.imTagDetect = readIm('invnoborder', strTag)

            self.ptsSymbolArea = getBoxCorners(self.symbolArea.tl[0], self.symbolArea.hw[0] )
            self.ptsDetectArea = getBoxCorners(self.detectArea.tl[0], self.detectArea.hw[0] )

            num = 2
            self.symbolSubAreas= self.symbolArea.getSubAreas(num, num)

            self.imSymbolSubAreas = []
            for area in self.symbolSubAreas:
                imSub = area.getRoi(self.imTagDetect)
                self.imSymbolSubAreas.append(imSub)

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

    def getSubAreas(self, row, col):
        # one cell dimensions
        hSub = int(self.hw[0] / row)
        wSub = int(self.hw[1] / col)

        # border pixels vertical
        hSubMulti = hSub * row
        if hSubMulti < self.hw[0]:
            # must append - > append to the last one
            hDiff = self.hw[0] - hSubMulti
        else:
            hDiff = 0

        # border pixels horizontal
        wSubMulti = wSub * col
        if wSubMulti < self.hw[1]:
            # must append - > append to the last one
            wDiff = self.hw[1] - wSubMulti
        else:
            wDiff = 0

        # create the subareas
        aSubs = []
        hw = (hSub, wSub)
        for iRow in range(0,row):
            for iCol in range(0,col):
                tl = (row*hSub, col*wSub)
                aSub = C_area( hw, tl)
                if iRow == row-1:
                    if iCol == col-1:
                        aSub = C_area( (hSub+hDiff,wSub+wDiff), tl)
                    else:
                        aSub = C_area( (hSub,wSub+wDiff), tl)
                if iCol == col-1:
                    aSub = C_area( (hSub,wSub+wDiff), tl)

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
    cTag = C_tag(strTag)
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

    mWarp = cSeenTag.mWarp

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