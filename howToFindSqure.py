import numpy as np
import cv2
from enum import Enum

from itertools import cycle



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

    corners_in_warped = findStableLineIntersection(cnt, self.external_contour_approx, plot=plot, half_interval=1)
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


def findClosestToMinAreaRect(im, mc, box, cnt):
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


import matplotlib.gridspec as gridspec

def findStableLineIntersection(cnt, external_contour_approx, plot = False, half_interval=1, ):
    """
    Find points from countour which have the biggest change in direction of three consecutive pixels
    """
    norm = cv2.NORM_L2SQR

    # if external_contour_approx != cv2.CHAIN_APPROX_SIMPLE:
    #     print('Cannot find contour direction drift from not continuous external contour')
    #     return None

    vy = []
    vx = []
    inv = []
    # circular list
    # print(len(cnt))
    count = len(cnt)
    cnt = np.float32(cnt)
    # direction = np.eye(1,1)

    if plot == True:
        print('creating figure')
        plt.figure(1)
        thismanager = plt.get_current_fig_manager()
        thismanager.window.wm_geometry("+0+0")
        plt.clf()
        sp = 910
        markers = ['x','o','+','s']
    # print('b')


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

        # shift of one
        if coef == 1:
            if len(inv) > 0:
                if inv[-1] == -1:
                    inv_index = len(inv)

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

    # shift of one - better for squares viewed from big angle
    inv[inv_index] = -1

    angles = np.arctan2(np.array(vy), np.array(vx)).tolist()

    if plot == True:
        print('plotting inv & angles atan2')
        sp += 1
        plt.subplot(sp)

        plt.plot( inv )
        plt.ylabel('inv ')
        plt.xlim([0,count])

        sp += 1
        plt.subplot(sp)
        ang = [angle[0]*180/np.pi for angle in angles]
        plt.plot(ang)
        plt.ylabel('angles atan2')
        plt.xlim([0,count])

    # # normalize
    # pihalf = np.pi / 2
    # angles = [angle[0] + pihalf for angle in angles]
    angles = [angle[0] for angle in angles]

    pi2 = 2 * np.pi
    for q in range(count):
        if inv[q] > 0:
            # angles[q] += np.pi
            angles[q] += pi2
            # print('less')



    if plot == True:
        print('plotting angles')
        sp += 1
        plt.subplot(sp)
        plt.plot([angle*180/np.pi for angle in angles])
        plt.ylabel('angles')
        plt.xlim([0,count])

    # we want raising positive angles - so derivation will be positive -> so we want to find minimum
    # todo check whether the angles are raising
    # not exactly discrete derivation
    derivate = [ angles[(q ) % count] - angles[(q-1) % count] for q in range(count)]
    # when subtracting 2pi -> 0.1 its negative -> we want it to be positive "and small"

    if plot == True:
        print('plotting derivate')
        sp += 1
        plt.subplot(sp)
        plt.plot(derivate)
        plt.ylabel('derivates not normalized')
        plt.xlim([0,count])


    # corner_indexes = np.argpartition(np.array(derivate), -4)
    # corner_indexes = np.partition(np.array(derivate), 4)[:4]

    # find one minimum and 3 maximums
    diff = np.array(derivate)
    sorted_indexes = np.argpartition(diff,-3)
    max_indexes = (sorted_indexes[-3:]).tolist()
    min_index = [np.argmin(diff)]
    # print(sorted_indexes)


    # arrange them against indexes
    corner_indexes = np.sort(min_index + max_indexes)

    # between these peaks find zero diff level interval
    side_intervals = [range(corner_indexes[q], corner_indexes[(q + 1) % 4] ) for q in range(4)]
    side_intervals[3] = range(corner_indexes[3] - count, corner_indexes[0])
    # print(side_intervals)

    # from those indexes get cnt points into 4 cnt_intervals
    sides = [[], [], [], []]
    diff_limit = np.deg2rad(5)
    diff_abs = np.abs(diff)
    # print(diff_limit)


    for q in range(4):
        for index in side_intervals[q]:
            # print(abs(diff[index]))
            if diff_abs[index] < diff_limit:
                sides[q].append(cnt[index][0])
                if plot == True:
                    # plt.scatter(index , 0, marker=markers[q])
                    plt.scatter(index % count, 0, marker=markers[q])

    if plot == True:
        print('plotting sorted')
        sorted = diff[sorted_indexes]
        sp += 1
        plt.subplot(sp)
        plt.plot(sorted)
        plt.ylabel('sorted')
        plt.xlim([0,count])

        sp +=1
        plt.subplot(sp)
        # plt.subplot2grid((9,1), (sp, 1), rowspan=1)
        plt.axis('equal')
        for q in range(4):
            for index in side_intervals[q]:
                plt.scatter(cnt[index % count][0][0], cnt[index % count][0][1], marker=markers[q])

    # fitLine for those 4 intervals
    if plot == True:
        print('counting fitLine for 4 intervals')


    if plot == True:
        plt.draw()
        plt.show()

    lines = []
    diff_limit = np.deg2rad(5)
    for side in sides:
        if len(side) > 1:
            [_vx, _vy, _x0, _y0] = cv2.fitLine(np.array(side), norm, 0, diff_limit, 0.01)
            lines.append(np.array([[_x0, _y0], [_x0 + _vx, _y0 + _vy]]))
        else:
            return None


    if plot == True:
        print('getting intersections')
    corners = []
    for q in range(4):
        corners.append(getIntersection(lines[q], lines[(q + 1) % 4]))
    # print(np.array(corners))
    # get intersection of those 4 lines

    if plot == True:
        plt.draw()
        plt.show()

    return corners

import sys
def getIntersection(line1, line2):
    s1 = np.array(line1[0])
    e1 = np.array(line1[1])

    s2 = np.array(line2[0])
    e2 = np.array(line2[1])

    a1 = (s1[1] - e1[1]) / (s1[0] - e1[0])
    b1 = s1[1] - (a1 * s1[0])

    a2 = (s2[1] - e2[1]) / (s2[0] - e2[0])
    b2 = s2[1] - (a2 * s2[0])

    if abs(a1 - a2) < sys.float_info.epsilon:
        return False

    x = (b2 - b1) / (a1 - a2)
    y = a1 * x + b1

    return [x[0], y[0]]

def findDirectionDrift(cnt, external_contour_approx, plot = False):
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
    # direction = np.eye(1,1)

    if plot == True:
        plt.figure(1)
        plt.clf()
        sp = 510


    for q in range(count):
        first = q - half_interval
        last = q + half_interval
        contour_segment = np.array( [cnt[k % count][0] for k in range(first, last + 1)] )
        [_vx, _vy, _, _] = cv2.fitLine(contour_segment, norm, 0, 0.1, 0.1)

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

    if plot == True:
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


    if plot == True:
        sp += 1
        plt.subplot(sp)
        plt.plot(angles)
        plt.ylabel('angles')

    # we want raising positive angles - so derivation will be positive -> so we want to find minimum
    # todo check whether the angles are raising
    # not exactly discrete derivation
    derivate = [ angles[(q ) % count] - angles[(q-1) % count] for q in range(count)]
    # when subtracting 2pi -> 0.1 its negative -> we want it to be positive "and small"
    if plot == True:
        sp += 1
        plt.subplot(sp)
        plt.plot(derivate)
        plt.ylabel('derivates not normalized')


    # corner_indexes = np.argpartition(np.array(derivate), -4)
    # corner_indexes = np.partition(np.array(derivate), 4)[:4]
    diff = np.array(derivate)
    sorted_indexes = np.argpartition(diff,-3)
    max_indexes = (sorted_indexes[-3:]).tolist()
    min_index = [np.argmin(diff)]
    # print(sorted_indexes)

    if plot == True:
        sorted = diff[sorted_indexes]
        sp += 1
        plt.subplot(sp)
        plt.plot(sorted)
        plt.ylabel('sorted')

    # find one minimum and 3 maximums
    corner_indexes = np.sort(min_index + max_indexes)

    # print(corner_indexes)
    #
    if plot == True:
        plt.show()
        plt.draw()


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
