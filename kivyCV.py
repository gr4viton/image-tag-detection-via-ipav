from kivy.app import App
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from kivy.graphics import Color, Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.properties import ObjectProperty, StringProperty
from kivy.config import Config

import timeit
import time
import threading
from multiprocessing import Process, Queue

# Config.set('postproc', 'retain_time', '50')

import cv2
import numpy as np

import findHomeography as fh
from thisCV import *

# class PongApp
# class OperationWidget(GridLayout):
    # image
    # label


def convert_to_texture(im):
    return convert_rgb_to_texture(fh.colorify(im))

def convert_rgb_to_texture(im_rgb):
    buf1 = cv2.flip(im_rgb, 0)
    buf = buf1.tostring()
    texture1 = Texture.create(size=(im_rgb.shape[1], im_rgb.shape[0]), colorfmt='bgr')
    texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return texture1

class Multicopter(GridLayout):
    gl_left = ObjectProperty()
    gl_middle = ObjectProperty()
    gl_right = ObjectProperty()
    img_webcam = ObjectProperty()
    img_tags = ObjectProperty()
    # txt_numFound = StringProperty()
    # str_num_found = StringProperty()
    sla_tags = ObjectProperty()
    sla_steps = ObjectProperty()

    def __init__(self, capture_control, findtag_control, **kwargs):
        # make sure we aren't overriding any important functionality
        super(Multicopter, self).__init__(**kwargs)
        self.capture_control = capture_control
        # self.root_toggle_findtag = findtag_control.find_tag
        self.root_toggle_findtag = None

    def toggle_findtag(self):
        self.root_toggle_findtag()


    def toggle_source_id(self):
        self.capture_control.toggle_source_id()

    def sla_steps_add_widgets(self,imList):
        diff = len(imList) - len(self.sla_steps.children)
        if diff > 0: # create widgets
            for num in range(0,np.abs(diff)):
                # self.sla_steps.add_widget(Image())
                # self.sla_steps.add_widget(Label(size_hint_y = '0.1'))
                self.sla_steps.add_widget(Label())
                print('added widget')
        else:
            for num in range(0,np.abs(diff)):
                self.sla_steps.remove_widget( self.sla_steps.children[-1])
                print('removed widget')

    def update_sla_steps(self,imList):

        if imList is not None:
            if len(imList) > 0:
                if len(imList) != len(self.sla_steps.children):
                    self.sla_steps_add_widgets(imList)
                else:
                    for (imItem, img_Child) in zip(imList, self.sla_steps.children):
                        imName = imItem[0]
                        im = np.uint8( imItem[1][0] )
                        # img_Child.text = imName


                        img_Child.texture = self.convert_to_texture( im.copy() )

                        break

            # imWhole = []
            # k = 1
            # for imItem in imList:
            #     if imWhole == []:
            #         imWhole = imItem[1]
            #     else:
            #         print(k)
            #         # [ imWhole.append(im)]
            #         # imWhole = fh.joinIm([ [imWhole], [imItem[1]] ],1)


            # if len(imWhole.shape) == 2:
            #     imWhole = cv2.cvtColor(imWhole, cv2.COLOR_GRAY2RGB)
            # for imItem in imList:
            #     str_name = imItem[0]
            #
            #     imEnclosed = imItem[1]
            #     im = imEnclosed[0]
            #
            #     imColor = fh.colorify(im[0])
            #     im = im.copy()
            #     # print(str_name)
            #     # print(im.shape)
            #     break

            # self.root.img_webcam.texture = self.convert_to_texture(imColor)

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

class FindTagControl():
    """
    Shared class to control findtag algorythm execution
    """

    def __init__(self, lock):
        self.cTag = None
        self.imSteps = None
        self.imTags = None
        self.findtag_running = False


class FindTagThread(threading.Thread):
    """
    Thread running find tag algorythm
    """
    def __init__(self, findtag_control, capture_control):
        threading.Thread.__init__(self)
        self.findtag_control = findtag_control
        self.capture_control = capture_control

    def run(self):
        while True:
            pass
            # frame = self.capture_control.get_frame()
            # imSteps, imTags = stepCV(frame,cTag)
            # self.findtag_control.imTags = imTags.copy()
            # self.findtag_control.imSteps = imSteps.copy()
            #do the math
            # copy
            # findtag_control





class CaptureControl():
    """
    Shared class to control source capture execution
    """
    frame = LockedNumpyArray( np.ones( (32,24,3,), np.uint8 ) * 255 )
    capturing = LockedValue(False)
    thread_running = LockedValue(False)

    def __init__(self):
        self.capture_lock = threading.Lock()
        self.capture = None
        self.source_id = 0
        self.init_capture()
        self.sleepTime = 0.01
        # self.thread = threading.Thread(target=self.capture_loop)
        self.thread = None
        self.thread = threading.Thread(target=self.capture_loop)

    def init_capture(self):
        self.capture_lock.acquire()
        self.capture = cv2.VideoCapture()
        self.capture_lock.release()

    def capture_frame(self):
        self.capture_lock.acquire()
        if self.capture.isOpened() != True:
            # self.open_capture()
            raise('Cannot read frame as the capture is not opened')
        else:
            ret, frame = self.capture.read()
        self.capture_lock.release()
        self.frame = frame
        print(self.frame.shape)

    # def get_frame(self):
    #     # self.capture()
    #     return self.frame # the property class LockedValue manages the locking

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
        # self.thread.start()
        self.thread_running = True

    def on_stop(self):
        self.capturing = False
        self.thread_running = False
        self.close_capture()

    def capture_loop(self):
        while self.thread_running:
            if self.capturing:
                self.capture_frame()
            print(time.time(),' - ', threading.current_thread().name)
            # threading.currentThread.
            time.sleep(self.sleepTime)
            # frames?





class multicopterApp(App):
    frame = []
    # running_findtag = False
    def build(self):
        # root.bind(size=self._update_rect, pos=self._update_rect)
        h = 700
        w = 1300
        Config.set('kivy', 'show_fps', 1)
        Config.set('graphics', 'window_state', 'maximized')
        Config.set('graphics', 'position', 'custom')
        Config.set('graphics', 'height', h)
        Config.set('graphics', 'width', w)
        Config.set('graphics', 'top', 15)
        Config.set('graphics', 'left', 4)

        # Config.set('graphics', 'fullscreen', 'fake')
        # Config.set('graphics', 'fullscreen', 1)

        self.init_threads()
        self.root = root = Multicopter(self.capture_control, None)
        self.build_opencv()
        return root

    def run(self):
        capture_control = CaptureControl()
        capture_control.start_capturing()
        # capture_thread = CaptureThread(capture_control)

        # capture_control.thread = threading.Thread(target=capture_control.capture_loop)
        # th1 = threading.Thread(target=capture_control.capture_loop)
        # th2 = threading.Thread(target=capture_control.capture_loop)

        capture_control.thread.start()
        # th1.start()
        # th2.start()

        super(multicopterApp, self).run()

    def init_threads(self):
        self.capture_control = CaptureControl()
        # self.capture_control.start_capturing()
        # self.capture_thread = CaptureThread(self.capture_control)
        # self.capture_thread.start()


        # self.capture_control.thread_running = True
        # self.capture_thread.run()

        # print('aaa')
        # # self.capture_thread.join()
        # self.capture_thread.join(timeout=0)
        # # self.capture_thread.run()
        print('aaa')

        # cap = cv2.VideoCapture(self.videoId)

    # def image_capture(self):


    def build_opencv(self):
        # self.source_id = 0
        # self.capture = cv2.VideoCapture(self.source_id)
        self.cTag = fh.readTag('2L')

        # ret, frame = self.capture.read()
        # if frame is None:
        #     raise('No camera attached on sourceId ' + str(self.source_id))

        # self.update_fps = 1.0/33.0
        self.fps_source = 1.0/50.0
        # self.fps_findTag = 1.0/1.0 # make it through trigger to play as fast as possible

        Clock.schedule_interval(self.update_source, self.fps_source )
        print('scheduled interval')
        # capturing thread

        # self.q_frame = Queue()
        #
        # d = dict( [ ('q_frame', self.q_frame ),
        #             ('running_findtag', self.running_findtag)] )
        # p = Process(target=self.image_capture, kwargs=d)
        #
        # # findtag algorythm thread
        # thread_findtag_dict = {}
        # thread_findtag_dict = dict( [('frame',self.frame),
        #                              ('cTag', self.cTag),
        #                              ('running_findtag', self.running_findtag)] )
        # print(thread_findtag_dict)
        # self.thread_findtag = threading.Thread(target=self.update_findtag,
        #                                 name='update_findtag', kwargs=thread_findtag_dict)
        # self.thread_findtag.start()
        # # self.thread_findtag.run()
        # self.set_running_findtag(True)


    # def set_running_findtag(self, new_value):
    #     if self.running_findtag != new_value:
    #         # Clock.schedule_interval(self.update_findtag, self.fps_findTag )
    #         self.running_findtag = new_value

    # def start_findtag(self):
    #     if self.running_findtag == False:
    #         # Clock.schedule_interval(self.update_findtag, self.fps_findTag )
    #         self.running_findtag = True
    #         print('Started findtag algorythm')
    #
    # def stop_findtag(self):
    #     if self.running_findtag == True:
    #         # Clock.unschedule(self.update_findtag)
    #         self.running_findtag = False
    #         print('Stopped findtag algorythm')

    # def toggle_findtag(self):
    #     self.set_running_findtag(not self.running_findtag)

    # def update_findtag(frame, cTag, running_findtag):
    #
    #     while True:
    #         if running_findtag:
    #
    #             start = timeit.timeit()
    #             # imList, imTags = stepCV(frame,self.cTag)
    #             imGray, imTags = stepCV(frame, cTag)
    #             # time.sleep(1)
    #             end = timeit.timeit()
    #             # print(end - start)
    #
    #             # self.root.img_webcam.texture = self.convert_to_texture(frame)
    #             # imList = imList.copy()
    #             # self.root.update_sla_steps(imList)
    #
    #
    #             # if len(self.root.sla_steps.children) == 0:
    #             #     self.root.sla_steps.add_widget(Image())
    #             # self.root.sla_steps.children[0].texture = convert_to_texture(imGray)
    #
    #             # if imTags is not None:
    #             #     # self.root.txt_numFound.text = str(len(imTags))
    #             #     if len(imTags) > 0:
    #             #         imAllTags = fh.joinIm( [[im] for im in imTags], 1 )
    #             #         # update_image(image_label, imAllTags)
    #             #         if len(imAllTags.shape) == 2:
    #             #             imAllTags = cv2.cvtColor(imAllTags, cv2.COLOR_GRAY2RGB)
    #             #         print(imAllTags)
    #             #         print(imAllTags.shape)
    #             #         self.root.img_tags.texture = self.convert_to_texture( imAllTags.copy() )
    #



    def update_source(self, dt):
        # ret, frame = self.capture.read()
        # print(self.capture_control.frame)
        self.frame = self.capture_control.frame
        if self.frame is not None:
            # print(self.frame.shape)
            self.root.img_webcam.texture = convert_to_texture(self.frame)

    def on_stop(self):
        print("Stopping capture")
        # self.capture.release()
        self.capture_control.on_stop()








if __name__ == '__main__':
    multicopterApp().run()



