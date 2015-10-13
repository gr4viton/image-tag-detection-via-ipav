import numpy as np
import cv2

class C_observedTag:
    # static class variable - known Tag
    # tagModels = loadTagModels('2L')
    def __init__(self, imTagInScene):
        self.imScene = imTagInScene
    # functions:

        # dst_pts, mWarp = fh.findWarpMatrix(imTagInScene, cTagModel)
    # find Warp etc
    # find square corners
    # etc..


def findWarpMatrix(imScene, cTag):

    src_pts = cTag.ptsDetectArea

    dst_pts = findSquare(imScene)
    mWarp, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # matchesMask = mask.ravel().tolist()
    # what is mask ?!
    # mWarp = addWarpRotation(mWarp, cTag, imScene)

    return dst_pts, mWarp

class C_tag: # tag model
    def __init__(self, strTag):
        # later have function to get this from actual image

        if strTag == '2L':
            hwWhole = 250
            bSymbolArea = 60
            bDetectArea = 40

            self.whole = C_area([hwWhole ]*2, [0]*2)
            b = bSymbolArea
            self.symbolArea = C_area([hwWhole - b*2],[b]*2)
            b = bDetectArea
            self.detectArea = C_area([hwWhole - b*2],[b]*2)

            self.imTag = readIm('full', strTag)
            self.imTagDetect = readIm('invnoborder', strTag)

            self.ptsSymbolArea = getBoxCorners(self.symbolArea.tl[0], self.symbolArea.hw[0] )
            self.ptsDetectArea = getBoxCorners(self.detectArea.tl[0], self.detectArea.hw[0] )
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
        #print(dot)

        pt = [int(dot[0]),int(dot[1])]
        # col = (255, 0, 0)
        col = 128
        cv2.circle(im, tuple(pt), 4, col, 1)
        if numbers == 1:
            # font = cv2.FONT_HERSHEY_SIMPLEX
            font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            cv2.putText(im,str(i), tuple([ d+10 for d in pt ]), font, 1.7, col, cv2.LINE_AA )
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
    #corner.append [mc, dist]
    corner_pts = np.float32(corner_pts)
    # print(corner_pts)


    distSq = [0] *4 # distance = distFromCenter - distFromMinBox
    distSq = np.float32(distSq)
    # print(distSq)

    cnt = np.float32(cnt)
    # print('starting to count')

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
    color = 150
    int_box = np.int0(corner_pts)
    cv2.drawContours(im,[int_box],0,color,1)
    return corner_pts

def findSquare(im):

    _, contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corner_pts = []
    for q in np.arange(len(contours)):
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
        # x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)

        # rotated boundingbox

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        # draw rotated minAreaRect boundingBox
        # color = 150
        # int_box = np.int0(box)
        # cv2.drawContours(im,[int_box],0,color,1)

        # leftmost etc
        # dst_pts.extend([cnt[cnt[:,:,0].argmin()][0]] ) # leftmost
        # dst_pts.extend([cnt[cnt[:,:,1].argmin()][0]] ) # topmost
        # dst_pts.extend([cnt[cnt[:,:,0].argmax()][0]] ) # rightmost
        # dst_pts.extend([cnt[cnt[:,:,1].argmax()][0]] ) # bottommost

        # corner_pts = findClosestToMinAreaRect(im,mc,box,cnt)
        corner_pts = findClosestToMinAreaRectAndFarthestFromCenter(im,mc,box,cnt)
        # corner_pts = findFarthestFromCenter(im,mc,box,cnt)
    if corner_pts == []:
        # return np.array( [[[0]*2]*4] )
        return np.array( [[0]*2,[1]*2,[2]*2,[42]*2] )
    return corner_pts

def matDot(A,B):
    C = np.eye(3,3)
    np.dot(A,B,C)
    return C


def addWarpRotation(imScene,imTag):

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


    mFinal = mInverse.copy()
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

def drawTagWarpedToScene(mWarp, imTag, imScene):
    h,w = imTag.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, mWarp)
    return cv2.polylines(imScene,[np.int32(dst)],True, 128,3, cv2.LINE_8)

def drawSceneWarpedToTag(mWarp, imScene, dims):
    return cv2.warpPerspective(imScene, mWarp, dims) #, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)



if __name__ == '__main__':

    strTag = '2L'
    # [imTag, ptTag] = readImTag('invnoborder', '2L')
    # imTag
    cTag = readTag(strTag)

    imScene = readIm('space1', strTag)
    imSceneOrig = imScene.copy()

    # im1 = rotate(im1,180)
    # imTag = rotate(imTag,90)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dst_pts, mWarp = findWarpMatrix(imScene,cTag)
    print(cTag.ptsDetectArea)
    print(dst_pts)
    imScene = drawDots(imScene,dst_pts)

    print(str(mWarp) + " = Homography transformation matrix")

    # get inverse transformation matrix
    mInverse = np.linalg.inv(mWarp)

    imTag = drawDots(cTag.imTagDetect.copy(), cTag.ptsDetectArea)
    im3 = drawTagWarpedToScene(mWarp, imTag, imScene)


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    mFinal = addWarpRotation(imScene,cTag.imTagDetect)

    imTagFromScene = cTag.imTagDetect.copy()
    imTagFromScene = drawSceneWarpedToTag(mInverse, imSceneOrig, cTag.imTagDetect.shape)

    im = imTagFromScene
#    im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


    imTagFromSceneRotated = imTagFromScene.copy()

    imTagFromSceneRotated = drawSceneWarpedToTag(mFinal, imSceneOrig, imTagFromScene.shape)
    # cv2.warpPerspective(imSceneOrig, mFinal, imTagFromScene.shape) #, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    # cv2.imshow('aa',im1copy)


    imBoth = joinIm([[imTagFromScene],[imTag],[imScene] ])
    # im3 =  cv2.cvtColor(im3 , cv2.COLOR_GRAY2RGB)

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



