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
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.config import Config

import cv2
import numpy as np
# import sys
import threading
import time

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

# class ImageButton(ButtonBehavior, Image):
#
#     # def __init__(self, toggle_drawing, **kwargs):
#         # super(Image, self).__init__(**kwargs)
#         # super(ButtonBehavior, self).__init__(**kwargs)
#         # self.toggle_drawing = toggle_drawing
#     pass

class StepWidget(GridLayout):

    name = StringProperty()
    drawing = ObjectProperty('down')
    kivy_image = ObjectProperty()
    toggle_object = ObjectProperty()
    # layout_steps_height = NumericProperty(1600)

    def __init__(self, **kwargs):
        super(StepWidget, self).__init__(**kwargs)
        # self.layout_steps = kwargs['parent']
        self.name = ''
        self.drawing = True
        self.texture = Texture.create(size = (42,42), colorfmt='bgr')
        self.name = 'default name'
        # self.kivy_image = ImageButton(self.toggle_drawing)
        # self.kivy_image = ImageButton()
        # self.add_widget(self.kivy_image)

    def recreate_texture(self, cv_image):
        self.texture = Texture.create(
            size=(cv_image.shape[1], cv_image.shape[0]), colorfmt='bgr')
        self.update_texture(cv_image)

    def recreate_widget(self, cv_image, name):
        self.recreate_texture(cv_image)
        self.name = name
        print('Recreated widget:',name,'\nwith dimensions:',cv_image.shape)

    def update_texture(self, im):
        # print(im.shape)
        if self.drawing: # called only if intended to draw
            self.update_texture_from_rgb(fh.colorify(im))

    def update_texture_from_rgb(self,im_rgb):
        buf1 = cv2.flip(im_rgb, 0)
        buf = buf1.tostring()
        self.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # print(im_rgb.shape)
        self.kivy_image.texture = self.texture

    def set_drawing(self, value):
        # print(value)
        self.drawing = value
        if value == True:
            self.toggle_object.state = 'down'
            # self.kivy_image.size_hint = (1, 0.9)
            # self.kivy_image.width = 100
            self.size_hint_x = 0.33
            # self.size_hint_y = 1
        if value == False:
            self.toggle_object.state = 'normal'
            # self.kivy_image.width = 40
            # self.kivy_image.size_hint = (0.1, 0.1)
            self.size_hint_x = 0.33/9
            # self.size_hint_y = 0.2

    def toggle_drawing(self):
        if self.toggle_object.state == 'down':
            self.set_drawing(False)
        else:
            self.set_drawing(True)


class StepWidgetControl():

    def __init__(self, layout_steps):
        self.layout_steps = layout_steps

    def show(self, command):
        if command == 'all':
            [widget.set_drawing(True) for widget in self.layout_steps.children]
        elif command == 'none':
            [widget.set_drawing(False) for widget in self.layout_steps.children]

    def layout_steps_add_widgets(self, im_steps):
        diff = len(im_steps) - len(self.layout_steps.children)
        if diff > 0: # create widgets
            for num in range(0, np.abs(diff)):
                self.layout_steps.add_widget(StepWidget())
                print('added widget')
        else:
            for num in range(0, np.abs(diff)):
                self.layout_steps.remove_widget( self.layout_steps.children[-1])
                print('removed widget')

        [widget.recreate_widget(np.uint8(im_item[1][0]), im_item[0])
         for (widget, im_item) in zip(self.layout_steps.children, im_steps)]

    def update_layout_steps(self, im_steps):

        if im_steps is not None:
            # im_steps = [im_steps[1],im_steps[1]]
            if len(im_steps) > 0:
                if len(im_steps) != len(self.layout_steps.children):
                    self.layout_steps_add_widgets(im_steps)
                else:
                    # print('hej')
                    [widget.update_texture(np.uint8(im_item[1][0].copy()))
                     for (im_item, widget) in zip(im_steps, self.layout_steps.children)]

def rgb_to_str(rgb):
    """Returns: string representation of RGB without alpha

    Parameter rgb: the color object to display
    Precondition: rgb is an RGB object"""
    return '[ '+str(rgb[0])+', '+str(rgb[1])+', '+str(rgb[2])+' ]'

class Multicopter(GridLayout):
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
    label_mean_exec_time = ObjectProperty()
    # img_tags_background = ObjectProperty((0.08, 0.16 , 0.24))
    grid_img_tags = ObjectProperty()

    def __init__(self, capture_control, findtag_control, **kwargs):
        # make sure we aren't overriding any important functionality
        super(Multicopter, self).__init__(**kwargs)
        self.capture_control = capture_control
        self.findtag_control = findtag_control

        self.step_widgets_control = StepWidgetControl(self.layout_steps)
class multicopterApp(App):
    # frame = []
    # running_findtag = False
    def build(self):
        # root.bind(size=self._update_rect, pos=self._update_rect)
        h = 700
        w = 1300
        Config.set('kivy', 'show_fps', 1)
        Config.set('kivy', 'desktop', 1)

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

        minHeight = 50
        print('Captured frame with dimensions',self.capture_control.frame.shape,
              '. Waiting until the heighth is greater than', minHeight, 'px')
        while self.capture_control.frame.shape[0] < minHeight:
            pass
        print('Captured frame with dimensions',self.capture_control.frame.shape,
              '. Continuing with program execution.')
        # time.sleep(0.5)

        self.findtag_control = FindtagControl(self.capture_control)
        self.findtag_control.start_findtagging()


        self.root = root = Multicopter(self.capture_control, self.findtag_control)

        self.build_opencv()

        # self.capture_control.toggle_source_id() # take the second input source
        return root

    def build_opencv(self):
        self.fps_redraw = 1.0/50.0
        self.fps_findtag = 1.0/50.0

        Clock.schedule_interval(self.redraw_capture, self.fps_redraw )
        print('Scheduled redraw_capture with fps = ', 1/self.fps_redraw)

        Clock.schedule_interval(self.redraw_findtag, self.fps_findtag )
        print('Scheduled redraw_findtag with fps = ', 1/self.fps_findtag)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # redraw_capture() create one texture object - dont create every time!

    # why is it so black?

    # step control to work
    # timeit individual steps and display on widgets
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def redraw_capture(self, dt):
        frame = self.capture_control.frame
        if frame is not None:
            self.root.img_webcam.texture = convert_to_texture(frame)

    def set_img_tags(self, found = False):
        if(found == False):
            self.root.grid_img_tags.color = (.08, .16 , .24)
        else:
            self.root.grid_img_tags.color = (.08, .96 , .24)

    def redraw_findtag(self, dt):
        # im_steps = self.findtag_control.im_steps
        # if im_steps is not None:
        #     self.root.img_steps.texture = convert_to_texture(im_steps)

        im_steps = self.findtag_control.im_steps
        self.root.step_widgets_control.update_layout_steps(im_steps)
        # self.root.update_layout_steps(im_steps)

        if len(self.findtag_control.execution_time) > 0:
            self.root.label_mean_exec_time.text = str(
                str(np.round(self.findtag_control.execution_time[-1], 5) * 1000) + "\n" +
                str(np.round(self.findtag_control.mean_execution_time, 5) * 1000)
                )
        imTags = self.findtag_control.im_tags

        if imTags is not None:
            self.root.txt_numFound.text = str(len(imTags))

            if len(imTags) > 0:
                self.set_img_tags(True)
                imAllTags = fh.joinIm( [[im] for im in imTags], 1 )
                if len(imAllTags.shape) == 2:
                    imAllTags = cv2.cvtColor(imAllTags, cv2.COLOR_GRAY2RGB)
                self.root.img_tags.texture = convert_to_texture( imAllTags.copy() )
            else:
                self.set_img_tags(False)
        else:
            self.set_img_tags(False)

    def on_stop(self):
        print("Stopping capture")
        self.capture_control.on_stop()
        self.findtag_control.on_stop()

if __name__ == '__main__':
    multicopterApp().run()



