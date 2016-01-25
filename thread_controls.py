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
            # ret_val = self.val.copy()
            ret_val = self.val
        else:
            ret_val = None
        self.lock.release()
        # print('getting', ret_val)
        return ret_val

    def __set__(self, obj, val):
        self.lock.acquire()
        # print('setting', val)
        # self.val = val.copy() # ????????????????????????????????????????? do i need a copy??
        self.val = val
        self.lock.release()

class FindtagControl():
    """
    Shared class to control findtag algorythm execution
    """


    step_control = LockedNumpyArray()
    seen_tags = LockedNumpyArray()
    findtagging = LockedValue(False)
    model_tag = LockedValue()
    # execution_time = LockedValue([])
    mean_execution_time = LockedValue(0)

    def __init__(self, capture_control, tag_names, selected_tag_name):
        self.tag_names = tag_names
        self.init_tag_names()
        self.select_tag_name(selected_tag_name)

        self.capture_control = capture_control
        self.execution_time_len = 50
        self.execution_time = []
        self.resolution_multiplier = 0.5
        self._step_control = StepControl(self.resolution_multiplier, self.model_tag)


    def init_tag_names(self):
        self.model_tags = [fh.read_model_tag(tag_name) for tag_name in self.tag_names]

    def select_tag_name(self, selected_tag_name):
        if selected_tag_name in self.tag_names:
            self.model_tag = self.model_tags[ self.tag_names.index(selected_tag_name)]
            self.selected_tag_name = selected_tag_name
        else:
            self.model_tag = self.model_tags[0]
            self.selected_tag_name = self.model_tags[0].strTag()




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
        if len(self.execution_time) > self.execution_time_len:
            self.execution_time.pop(0)
            self.add_exec_times(tim)
        else:
            self.execution_time.append(tim)
        self.mean_execution_time = np.sum(self.execution_time) / len(self.execution_time)

    def findtag_loop(self):
        while self.findtagging:
            self.findtag()

    def findtag(self):
        start = time.time()
        self._step_control.step_all(self.capture_control.frame, self.resolution_multiplier )
        end = time.time()
        self.add_exec_times(end-start)

        # not thread safe
        self.step_control = self._step_control
        self.seen_tags = self._step_control.seen_tags

        # here raise an event for the conversion and redrawing to happen
        # time.sleep(0.0001)


    def update_findtag_gui(self, frame, tag_model, running_findtag):

        while True:
            if running_findtag:
                self.findtag()

    def set_resolution_div(self, resolution_multiplier):
        self.resolution_multiplier = resolution_multiplier

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