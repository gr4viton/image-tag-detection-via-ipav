import timeit
import time
import threading

from multiprocessing import Queue

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
        self.val = val.copy() # ????????????????????????????????????????? do i need a copy??
        self.lock.release()

class FindtagControl():
    """
    Shared class to control findtag algorythm execution
    """

    im_steps = LockedNumpyArray()
    im_tags = LockedNumpyArray()
    findtagging = LockedValue(False)
    model_tag = LockedValue()
    # exec_times = LockedValue([])
    mean_exec_time = LockedValue(0)

    def __init__(self, capture_control):
        self.capture_control = capture_control
        self.init_findtag()
        self.exec_times_max_len = 50
        self.exec_times = []

    def init_findtag(self):
        self.model_tag = fh.read_model_tag('2L')

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

    def add_exec_times(self, tim):
        if len(self.exec_times) > self.exec_times_max_len:
            self.exec_times.pop(0)
            self.add_exec_times(tim)
        else:
            self.exec_times.append(tim)
        self.mean_exec_time = np.sum(self.exec_times) / len(self.exec_times)

    def findtag_loop(self):
        while self.findtagging:
            self.findtag()

    def findtag(self):
        start = timeit.timeit()
        im_steps, im_tags = stepCV(self.capture_control.frame, self.model_tag)
        end = timeit.timeit()
        self.add_exec_times(end-start)
        # print(end - start)
        # print(im_gray.shape)
        self.im_steps = im_steps
        self.im_tags = im_tags

        # here raise an event for the conversion and redrawing to happen
        # time.sleep(0.0001)


    def update_findtag_gui(self,frame, tag_model, running_findtag):

        while True:
            if running_findtag:
                self.findtag()

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
        #self.source_id = np.mod(self.source_id+1, 2)
        # self.open_capture()
        try_next = True
        while try_next:
            self.source_id += 1
            self.capture.open(self.source_id)
            if self.capture.isOpened() != True:
                print('Cannot open capture source_id ', self.source_id)
                self.source_id = -1
                continue
            ret, frame = self.capture.read()
            if ret == False:
                print('Source cannot be read from, source_id ', self.source_id)
                self.source_id = -1
                continue
            print('Opened capture source_id ' + str(self.source_id))
            try_next = False
        self.capture_lock.release()

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