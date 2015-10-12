import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as mcm
import matplotlib as mpl

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out
def orbDesc(im1,im2, imBoth):

    # Initiate ORB detector
    # orb = cv2.ORB_create( nfeatures = 300 )
    orb = cv2.ORB_create( nfeatures = 400 )

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(im1,None)
    kp2, des2 = orb.detectAndCompute(im2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch( des1,des2, k=2)

    # Apply ratio test
    good = []
    goodIdx = []
    idx = 0
    const = 0.80
    # const = 0.90

    for m,n in matches:
        if m.distance < const*n.distance:
            good.append([m])
            goodIdx.append(idx)
        idx += 1

    # img3 = []

    # change kp points - useless i will not use knnMatch and orb..
    # for m in goodIdx:
    #     # kp1[m].pt = tuple(x+100 for x in kp1[m].pt)
    #     print(m)
    #     print(kp2[m].pt)
    #     kp2[m].pt = tuple(x*0.5+100 for x in kp2[m].pt)
    #     # kp2[m].pt = tuple([10*m, 10])
    #     print(kp2[m].pt)
        # kp1[m].pt += 10.0


    # Draw first 10 matches.
    # cv2.drawMatchesKnn(img1,kp1,img2,kp2, matches[:10], outImg=img3, flags=2)
    cv2.drawMatchesKnn(im1,kp1,im2,kp2, good, outImg=imBoth, flags=2)


    # img3 = drawMatches(img1,kp1,img2,kp2,matches[:20])

    # vytvor si dvojice sam - ze znalosti rohu ctverce

    print(str(len(des1)) + " = number of tag descriptor")
    print(str(len(des2)) + " = number of descriptor in scene image" )
    print(str(len(matches)) + "= all matches" )
    print(str(len(good)) + " = good matches")

    # print(goodIdx)
    # MIN_MATCH_COUNT = 4
    #
    # if len(good)>MIN_MATCH_COUNT:
    #     # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    #     # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #
    #     src_pts = np.float32([ kp1[m].pt for m in goodIdx ]).reshape(-1,1,2)
    #     dst_pts = np.float32([ kp2[m].pt for m in goodIdx ]).reshape(-1,1,2)
    #
    # else:
    #     print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    #     matchesMask = None
    #     exit

    return imBoth

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
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


class C_tag:
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
    # side = 250
    # bSize = 40 # from corner to corner of the inner square
    # innerSize = side - 2 * bSize # inner square side
    # bInnerSize = 60 # border from corner of the inner to the corner of the symbol (60 for tag2)
    # #sizAwidth = sizBwidth
    # #sizAleft =  sizBleft

def getBoxCorners(boxOffset, boxSide):
    aS = boxOffset
    aB = boxOffset + boxSide
    src_pts = []
    pts = [[aS, aS], [aS, aB], [aB, aB], [aB, aS]]
    # [src_pts.append(pt) for pt in pts]
    return np.float32(pts)


def readTag(strTag):
    cTag = C_tag(strTag)
    return cTag



def makeBorder(im, bgColor):
    # rect = [0, 0, im.shape[0], im.shape[1]]
    bs = max(im.shape)
    im = cv2.copyMakeBorder(im, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=bgColor)
    return im, bs

def makeLaplacian(im):
    return np.uint8(np.absolute(cv2.Laplacian(im, cv2.CV_64F)))

def gaussIt(im):
    a = 15
    # return cv2.blur(im,(a,a))
    a = 75
    return cv2.bilateralFilter(im,9,a,a)

def joinIm(im1,im2):
    hDiff = im2.shape[0] - im1.shape[0]
    im1a = cv2.copyMakeBorder(im1,0,hDiff,0,0,cv2.BORDER_CONSTANT, value=0)

    imBoth = np.hstack([im1a,im2])
    return imBoth

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


def findWarpMatrix(imScene, cTag):

    src_pts = cTag.ptsDetectArea

    dst_pts = findSquare(imScene)
    mWarp, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # matchesMask = mask.ravel().tolist()
    # what is mask ?!


    return dst_pts, mWarp


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

    imBoth = joinIm(im3,joinIm(imTagFromScene,joinIm(imTag,imScene)))
    # im3 =  cv2.cvtColor(im3 , cv2.COLOR_GRAY2RGB)
    # imAll = np.hstack([imBoth,im3])
    imAll = colorify(imBoth)

    # a = 0.5
    # imAll = cv2.resize(imAll, (0, 0), fx=a, fy=a)

    cv2.imshow('images',imAll)
    # plt.imshow(imgAll, cmap = mpl.cm.Greys_r)
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

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

