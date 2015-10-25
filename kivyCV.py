from kivy.app import App
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from kivy.graphics import Color, Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.properties import ObjectProperty, StringProperty

import cv2
import numpy as np

import findHomeography as fh
from thisCV import *

# class PongApp

class Multicopter(GridLayout):
    gl_left = ObjectProperty()
    gl_middle = ObjectProperty()
    gl_right = ObjectProperty()
    img_webcam = ObjectProperty()
    img_tags = ObjectProperty()
    txt_numFound = ObjectProperty()
    # info = StringProperty()

    def __init__(self, toggle_source_id, **kwargs):
        # make sure we aren't overriding any important functionality
        super(Multicopter, self).__init__(**kwargs)
        self.root_toggle_source_id = toggle_source_id

    def toggle_source_id(self):
        self.root_toggle_source_id()
class multicopterApp(App):
    def build(self):
        self.root = root = Multicopter(self.toggle_source_id)
        # root.bind(size=self._update_rect, pos=self._update_rect)

        self.build_opencv()

        return root

    def build_opencv(self):
        self.source_id = 0
        self.capture = cv2.VideoCapture(self.source_id)
        self.cTag = fh.readTag('2L')

        ret, frame = self.capture.read()
        if frame is None:
            raise('No camera attached on sourceId ' + str(self.source_id))
        self.update_fps = 1.0/33.0
        Clock.schedule_interval(self.update, self.update_fps )


    def toggle_source_id(self):
        Clock.unschedule(self.update)
        self.capture.release()
        self.source_id = np.mod(self.source_id+1, 2)
        print('New sourceId = ' + str(self.source_id))
        self.capture = cv2.VideoCapture(self.source_id)
        Clock.schedule_interval(self.update, self.update_fps )


    def update(self, dt):
        ret, frame = self.capture.read()
        imWhole, imTags = stepCV(frame,self.cTag)

        # self.root.img_webcam.texture = self.convert_to_texture(frame)

        if imWhole is not None:
            if len(imWhole.shape) == 2:
                imWhole = cv2.cvtColor(imWhole, cv2.COLOR_GRAY2RGB)
            self.root.img_webcam.texture = self.convert_to_texture(imWhole)
        if imTags is not None:
            txt_numFound = len(imTags)
            if len(imTags) > 0:
                imAllTags = fh.joinIm( [[im] for im in imTags], 1 )
                # update_image(image_label, imAllTags)
                if len(imAllTags.shape) == 2:
                    imAllTags = cv2.cvtColor(imAllTags, cv2.COLOR_GRAY2RGB)
                print(imAllTags)
                print(imAllTags.shape)
                self.root.img_tags.texture = self.convert_to_texture( imAllTags.copy() )

    def convert_to_texture(self, im):
        buf1 = cv2.flip(im, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(im.shape[1], im.shape[0]), colorfmt='bgr')
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        return texture1
    def on_stop(self):
        print("Stopping capture")
        self.capture.release()
if __name__ == '__main__':
    multicopterApp().run()



