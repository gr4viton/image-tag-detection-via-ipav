import timeit
import time
import threading

import cv2
import numpy as np

import findHomeography as fh

from thisCV import *

class LockedValue(object):
    """
    Thread safe numpy array
    """
    def __init__(self, init_val = None):
        self.lock = threading.Lock()
        self.val = init_val

    def __get__(self, obj, objtype):
        self.lock.acquire()
        if self.val != None:
            ret_val = self.val
        else:
            ret_val = None
        self.lock.release()
        # print('getting', ret_val)
        return ret_val

    def __set__(self, obj, val):
        self.lock.acquire()
        # print('setting', val)
        self.val = val
        self.lock.release()

class LockedNumpyArray(object):
    """
    Thread safe numpy array
    """
    def __init__(self, init_val = None):
        self.lock = threading.Lock()
        self.val = init_val

    def __get__(self, obj, objtype):
        self.lock.acquire()
        if self.val != None:
            ret_val = self.val.copy()
        else:
            ret_val = None
        self.lock.release()
        # print('getting', ret_val)
        return ret_val

    def __set__(self, obj, val):
        self.lock.acquire()
        # print('setting', val)
        self.val = val.copy() # ????????????????????????????????????????? do i need it?
        self.lock.release()

class FindtagControl():
    """
    Shared class to control findtag algorythm execution
    """

    im_steps = LockedNumpyArray()
    im_tags = LockedNumpyArray()
    findtagging = LockedValue(False)
    cTag = LockedValue()

    def __init__(self, capture_control):


        self.capture_control = capture_control
        # self.capture_lock = threading.Lock()
        # self.capture = None
        # self.source_id = 0

        self.init_findtag()

        self.sleepTime = 0.0

    def init_findtag(self):
        # load cTag
        self.cTag = fh.readTag('2L')
        pass
    #     self.capture_lock.acquire()
    #     self.capture = cv2.VideoCapture()
    #     self.capture_lock.release()
    #
    # def open_source_id(self, new_source_id):
    #     self.capture_lock.acquire()
    #     self.source_id = new_source_id
    #     self.capture_lock.release()
    #     self.open_source_id()
    #
    # def open_capture(self):
    #     self.capture_lock.acquire()
    #     self.capture.open(self.source_id)
    #     if self.capture.isOpened() != True:
    #         raise('Cannot open capture source_id ', self.source_id)
    #     print('Opened capture source_id ' + str(self.source_id))
    #     self.capture_lock.release()
    #
    # def toggle_source_id(self):
    #     self.capture_lock.acquire()
    #     self.source_id = np.mod(self.source_id+1, 2)
    #     self.capture_lock.release()
    #     self.open_capture()
    #
    # def close_capture(self):
    #     self.capture_lock.acquire()
    #     self.capture.release()
    #     self.capture_lock.release()
    #
    def start_findtagging(self):
        self.findtagging = True
        self.thread = threading.Thread(target=self.findtag_loop)
        self.thread.start()

    def toggle_findtagging(self):
        if self.findtagging == False:
            self.start_findtagging()
        else:
            self.findtagging = False

    def on_stop(self):
        self.findtagging = False

    def findtag_loop(self):
        while self.findtagging:
            self.findtag()

    def findtag(self):

        # start = timeit.timeit()
        imGray, imTags = stepCV(self.capture_control.frame, self.cTag)
        # imList, imTags = stepCV(self.capture_control.frame, self.cTag)
        # end = timeit.timeit()
        # print(end - start)


        # self.capture_lock.acquire()
        # if self.capture.isOpened() != True:
        #     # self.open_capture()
        #     raise('Cannot read frame as the capture is not opened')
        # else:
        #     ret, frame = self.capture.read()
        # self.capture_lock.release()
        # self.frame = frame
        # # print(self.frame.shape)

        # here raise an event for the conversion and redrawing to happen


    def update_findtag_gui(self,frame, cTag, running_findtag):

        while True:
            if running_findtag:

                imGray, imTags = stepCV(frame, cTag)
                # time.sleep(1)

                # self.root.img_webcam.texture = self.convert_to_texture(frame)
                # imList = imList.copy()
                # self.root.update_sla_steps(imList)


                # if len(self.root.sla_steps.children) == 0:
                #     self.root.sla_steps.add_widget(Image())
                # self.root.sla_steps.children[0].texture = convert_to_texture(imGray)

                # if imTags is not None:
                #     # self.root.txt_numFound.text = str(len(imTags))
                #     if len(imTags) > 0:
                #         imAllTags = fh.joinIm( [[im] for im in imTags], 1 )
                #         # update_image(image_label, imAllTags)
                #         if len(imAllTags.shape) == 2:
                #             imAllTags = cv2.cvtColor(imAllTags, cv2.COLOR_GRAY2RGB)
                #         print(imAllTags)
                #         print(imAllTags.shape)
                #         self.root.img_tags.texture = self.convert_to_texture( imAllTags.copy() )


class CaptureControl():
    """
    Shared class to control source capture execution
    """
    frame = LockedNumpyArray( np.ones( (32,24,3,), np.uint8 ) * 255 )
    capturing = LockedValue(False)

    def __init__(self):
        self.capture_lock = threading.Lock()
        self.capture = None

        self.init_capture()

        self.source_id = 0

        self.sleepTime = 0.0

    def init_capture(self):
        self.capture_lock.acquire()
        self.capture = cv2.VideoCapture()
        self.capture_lock.release()

    def open_source_id(self, new_source_id):
        self.capture_lock.acquire()
        self.source_id = new_source_id
        self.capture_lock.release()
        self.open_source_id()

    def open_capture(self):
        self.capture_lock.acquire()
        self.capture.open(self.source_id)
        if self.capture.isOpened() != True:
            raise('Cannot open capture source_id ', self.source_id)
        print('Opened capture source_id ' + str(self.source_id))
        self.capture_lock.release()

    def toggle_source_id(self):
        self.capture_lock.acquire()
        self.source_id = np.mod(self.source_id+1, 2)
        self.capture_lock.release()
        self.open_capture()

    def close_capture(self):
        self.capture_lock.acquire()
        self.capture.release()
        self.capture_lock.release()

    def start_capturing(self):
        self.open_capture()
        self.capturing = True
        self.thread = threading.Thread(target=self.capture_loop)
        self.thread.start()

    def toggle_capturing(self):
        if self.capturing == False:
            self.start_capturing()
        else:
            self.capturing = False

    def on_stop(self):
        # stops capturing and releases capture object
        self.capturing = False
        self.close_capture()

    def capture_loop(self):
        while self.capturing:
            self.capture_frame()
            time.sleep(self.sleepTime)  # frames?

    def capture_frame(self):
        self.capture_lock.acquire()
        if self.capture.isOpened() != True:
            # self.open_capture()
            raise('Cannot read frame as the capture is not opened')
        else:
            ret, frame = self.capture.read()
        self.capture_lock.release()
        self.frame = frame
        # print(self.frame.shape)