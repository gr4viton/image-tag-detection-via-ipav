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
# from kivy.uix.checkbox import CheckBox
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import ObjectProperty, StringProperty
from kivy.config import Config

import cv2
import numpy as np

import findHomeography as fh
from thisCV import *

from thread_controls import CaptureControl, FindtagControl


def convert_to_texture(im):
    return convert_rgb_to_texture(fh.colorify(im))

def convert_rgb_to_texture(im_rgb):
    buf1 = cv2.flip(im_rgb, 0)
    buf = buf1.tostring()
    texture1 = Texture.create(size=(im_rgb.shape[1], im_rgb.shape[0]), colorfmt='bgr')
    texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
    return texture1

class StepWidget():
    def __init__(self, im):
        # self.function
        self.texture = Texture.create(size=(im.shape[1], im.shape[0]), colorfmt='bgr')
        self.update_texture(im) # called only if intended to draw


    def update_texture(self,im):
        return convert_rgb_to_texture(fh.colorify(im))

    def update_texture_from_rgb(self,im_rgb):
        buf1 = cv2.flip(im_rgb, 0)
        buf = buf1.tostring()
        self.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')


class Multicopter(BoxLayout):
    gl_left = ObjectProperty()
    gl_middle = ObjectProperty()
    gl_right = ObjectProperty()
    img_webcam = ObjectProperty()
    img_tags = ObjectProperty()
    # txt_numFound = StringProperty()
    # str_num_found = StringProperty()
    sla_tags = ObjectProperty()
    layout_steps = ObjectProperty()
    # img_steps = ObjectProperty()

    def __init__(self, capture_control, findtag_control, **kwargs):
        # make sure we aren't overriding any important functionality
        super(Multicopter, self).__init__(**kwargs)
        self.capture_control = capture_control
        self.findtag_control = findtag_control

        self.step_widgets = []

    def layout_steps_add_widgets(self,im_list):
        diff = len(im_list) - len(self.layout_steps.children)
        if diff > 0: # create widgets
            for num in range(0, np.abs(diff)):
                self.layout_steps.add_widget(Image(size_hint_x = '0.1'))
                # self.layout_steps.add_widget(Label(size_hint_y = '0.1'))
                # self.layout_steps.add_widget(Button())
                # self.layout_steps.add_widget(ToggleButton(size_hint_x = '0.1'))
                print('added widget')
        else:
            for num in range(0, np.abs(diff)):
                self.layout_steps.remove_widget( self.layout_steps.children[-1])
                print('removed widget')

        # len_im_list = len(im_list)
        # diff = len_im_list - len(self.step_widgets)
        # if diff > 0: # create widgets
        #     for num in range(0, np.abs(diff)):
        #
        #         im = np.uint8(  (im_list[len_im_list - diff+num-1])[1] )
        #         print(len(im))
        #         self.step_widgets.append(StepWidget(im))
        #         print('added widget')
        # else:
        #     for num in range(0, np.abs(diff)):
        #         self.step_widgets.pop(-1)
        #         print('removed widget')

    def update_layout_steps(self,im_steps):

        if im_steps is not None:
            # im_steps = [im_steps[1],im_steps[1]]
            if len(im_steps) > 0:
                if len(im_steps) != len(self.layout_steps.children):
                    self.layout_steps_add_widgets(im_steps)
                else:
                    # print(len(self.layout_steps.children), ' = ', len(im_steps))
                    # for img_Child in self.layout_steps.children:

                    # for (imItem, img_Child, step_widget) in \
                    #         zip(im_steps, self.layout_steps.children, self.step_widgets):
                    for (imItem, img_Child) in zip(im_steps, self.layout_steps.children):
                        # print(imItem[0])
                        imName = imItem[0]
                        # imName = imItem[0][0]
                        im = np.uint8( imItem[1][0] )
                        img_Child.text = imName

                        # step_widget.update_texture(im[0])
                        # img_Child.texture = step_widget.texture

                        img_Child.texture = convert_to_texture( im.copy() )
                        img_Child.texture
                        # break

            # imWhole = []
            # k = 1
            # for imItem in im_steps:
            #     if imWhole == []:
            #         imWhole = imItem[1]
            #     else:
            #         print(k)
            #         # [ imWhole.append(im)]
            #         # imWhole = fh.joinIm([ [imWhole], [imItem[1]] ],1)


            # if len(imWhole.shape) == 2:
            #     imWhole = cv2.cvtColor(imWhole, cv2.COLOR_GRAY2RGB)
            # for imItem in im_steps:
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


class multicopterApp(App):
    frame = []
    # running_findtag = False
    def build(self):
        # root.bind(size=self._update_rect, pos=self._update_rect)
        h = 700
        w = 1300
        Config.set('kivy', 'show_fps', 1)
        # Config.set('graphics', 'window_state', 'maximized')
        Config.set('graphics', 'position', 'custom')
        Config.set('graphics', 'height', h)
        Config.set('graphics', 'width', w)
        Config.set('graphics', 'top', 15)
        Config.set('graphics', 'left', 4)
        Config.set('graphics', 'multisamples', 0) # to correct bug from kivy 1.9.1 - https://github.com/kivy/kivy/issues/3576

        # Config.set('graphics', 'fullscreen', 'fake')
        # Config.set('graphics', 'fullscreen', 1)

        self.capture_control = CaptureControl()
        self.capture_control.start_capturing()

        self.findtag_control = FindtagControl(self.capture_control)
        self.findtag_control.start_findtagging()

        self.root = root = Multicopter(self.capture_control, self.findtag_control)
        self.build_opencv()
        self.capture_control.toggle_source_id()
        return root

    def build_opencv(self):
        self.fps_redraw = 1.0/50.0
        self.fps_findtag = 1.0/50.0

        Clock.schedule_interval(self.redraw_capture, self.fps_redraw )
        print('Scheduled redraw_capture with fps = ', 1/self.fps_redraw)

        Clock.schedule_interval(self.redraw_findtag, self.fps_findtag )
        print('Scheduled redraw_findtag with fps = ', 1/self.fps_findtag)

    def redraw_capture(self, dt):
        frame = self.capture_control.frame
        if frame is not None:
            self.root.img_webcam.texture = convert_to_texture(frame)

    def redraw_findtag(self, dt):
        # im_steps = self.findtag_control.im_steps
        # if im_steps is not None:
        #     self.root.img_steps.texture = convert_to_texture(im_steps)

        im_steps = self.findtag_control.im_steps
        self.root.update_layout_steps(im_steps)

        imTags = self.findtag_control.im_tags

        # if len(self.root.layout_steps.children) == 0:
        #     self.root.layout_steps.add_widget(Image())
        # self.root.layout_steps.children[0].texture = convert_to_texture(imGray)

        if imTags is not None:
            self.root.txt_numFound.text = str(len(imTags))

            if len(imTags) > 0:
                imAllTags = fh.joinIm( [[im] for im in imTags], 0 )
                # update_image(image_label, imAllTags)
                if len(imAllTags.shape) == 2:
                    imAllTags = cv2.cvtColor(imAllTags, cv2.COLOR_GRAY2RGB)
                print(imAllTags)
                print(imAllTags.shape)
                self.root.img_tags.texture = convert_to_texture( imAllTags.copy() )


    def on_stop(self):
        print("Stopping capture")
        self.capture_control.on_stop()
        self.findtag_control.on_stop()

if __name__ == '__main__':
    multicopterApp().run()



