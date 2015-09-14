import numpy as np
import cv2

from collections import Counter

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### imclearborder definition

def imclearborder(imgBW, radius):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    #contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
    
    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

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
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        color = 0
        cv2.drawContours(imgBWcopy, contours, idx, color, -1)

    return imgBWcopy

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### makepairs(im)

def makepairs(imgBW):

    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _, contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    #contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
    
    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # list of pair-lists

    
    # For each contour...
    for q in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[q]
##        print "cnt"
##        print cnt[0,0]
        mu = cv2.moments(cnt)
        leftTop = tuple(cnt[0,0])
##        print mu
        if mu['m00'] == 0:
            continue
        mc = (
            int( mu['m10']/mu['m00']) ,
            int( mu['m01']/mu['m00']) )
##        print mc
        color = 128
        cv2.circle( imgBWcopy, leftTop, 4, color, -1, 8, 0 )
        color = 200
        cv2.circle( imgBWcopy, mc, 4, color, -1, 8, 0 )
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(imgBWcopy,(x,y),(x+w,y+h), color,2)
        
##        for k in leng :
##            if k != q:
##                if cnt

##        # Look at each point in the contour
##        for pt in cnt:
##            rowCnt = pt[0][1]
##            colCnt = pt[0][0]
##
##            # If this is within the radius of the border
##            # this contour goes bye bye!
##            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
##            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)
##
##            if check1 or check2:
##                contourList.append(idx)
##                break

##    for idx in contourList:
##        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### bwareaopen definition
def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            color = 0
            cv2.drawContours(imgBWcopy, contours, idx, color, -1)

    return imgBWcopy

def update(i):
    tbValue = i
    print i
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def inverte(im):
    return (255-im)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def stepCV(cap):
            #tbValue = cv2.getTrackbarPos('trackMe','image')
        # Capture frame-by-frame
        flag, frame = cap.read()
##        _, frame = cap.read()

        if flag==0:
            return None
        im = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    ##    im = cv2.resize(frame, (0,0), fx=0.7, fy=0.7)
    ##    im = cv2.resize(frame, (0,0), fx=0.1, fy=0.1)
    ##    im = frame

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = gray
        #ims.append(im) # gray

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ## gaussian blur
        blur = cv2.GaussianBlur(im,(5,5),0)
    ##    blur = cv2.GaussianBlur(im,(3,3),0)

        im = blur

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ## THRESH
        ret,th1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY_INV )
        th2 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)

        titles = ['Original Image', 'Global Thresholding (v = 127)',
                    'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## OTSU
        im = blur
        # find normalized_histogram, and its cumulative distribution function
        hist = cv2.calcHist([im],[0],None,[256],[0,256])
        hist_norm = hist.ravel()/hist.max()
        Q = hist_norm.cumsum()

        bins = np.arange(256)

        fn_min = np.inf
        thresh = -1

        for i in xrange(1,256):
            p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
            q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
            b1,b2 = np.hsplit(bins,[i]) # weights

            # finding means and variances
            m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
            v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

            # calculates the minimization function
            fn = v1*q1 + v2*q2
            if fn < fn_min:
                fn_min = fn
                thresh = i

        # find otsu's threshold value with OpenCV function
        ret, otsu = cv2.threshold(im, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        #ret, otsu = cv2.threshold(im,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #r,c = otsu.shape

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ##

        im = otsu
    ##    im = th1
    ##    im = inverte(th2)
    ##    im = inverte(th3)
        thresh = im
        killerBorder = 5
        clear = imclearborder(im, killerBorder)
        a = 5
        clear = cv2.copyMakeBorder(clear[a:-a,a:-a], a,a,a,a, cv2.BORDER_CONSTANT, value=0)
        #cv2.findContours(otsu

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # bwareaopen?

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # imfill
        im = clear
        flooded = im.copy()

        h, w = im.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        mask[:] = 0
        seed = None

        cv2.floodFill(flooded, mask, seed, 0, 0,255, 4 )

    ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # makepairs
        im = flooded
        paired = makepairs(im)

        #ims.append([th1, th2, th3])

        #ims = [gray, th1, th2, th3, blur, otsu]
    ##    ims = [gray, otsu]

    ##    ims = [gray, blur, otsu, clear, flooded]
        ims = [gray, thresh, clear, flooded, paired]
        imWhole = np.vstack(ims)
    #    cv2.imshow('imColor', frame)
        return imWhole
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def loopCV(cap):
    print "loopCV started"
    while(True):
        im = stepCV(cap)
        cv2.imshow('image', im )
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


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#### Main program

# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
 
#C:\PROG\dev\opencv\opencv\sources\data\haarcascades\

if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0)
    ims = []

    cv2.namedWindow('image')

    tbValue = 3
    maxValue = 6
    cv2.createTrackbar( "trackMe", "image", tbValue, maxValue, update )
    loopCV(cap)
    cap.release()
    cv2.destroyAllWindows()

