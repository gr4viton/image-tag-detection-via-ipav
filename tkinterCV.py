#!/usr/bin/python

#http://stackoverflow.com/questions/17073227/display-an-opencv-video-in-tkinter-using-multiprocessing

import numpy as np
from multiprocessing import Process, Queue
from Queue import Empty
import cv2
#import cv2.cv as cv
from PIL import Image, ImageTk
import time
import Tkinter as tk
from thisCV import *

# import sys

#tkinter GUI functions----------------------------------------------------------
def quit_(root, process, *whatever):
   process.terminate()
   root.destroy()
# def quitCallback():

def update_image(image_label, queue):
   frame = queue.get()
##   im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   im = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
   a = Image.fromarray(im)
   b = ImageTk.PhotoImage(image=a)
   image_label.configure(image=b)
   image_label._image_cache = b  # avoid garbage collection
   root.update()

def update_all(root, image_label, queue):
   # print "updating all"
   update_image(image_label, queue)
   # print "updated all"
   root.after(0, func=lambda: update_all(root, image_label, queue))

#multiprocessing image processing functions-------------------------------------
def image_capture(queue):
   cap = cv2.VideoCapture(0)
   loopingCV = 1
   while loopingCV:
      queue.put(stepCV(cap))
   cap.release()


def GUI_setup(root):
   # GUI Items
   print 'GUI initialized...'
   image_label = tk.Label(master=root)# label for the video frame

   print 'GUI image label initialized...'
   # quit button
   quit_button = tk.Button(master=root, text='Quit',command=lambda: quit_(root,p))

   txHell = tk.Entry(master=root, text='tkInter back in town')
##   lbHell = tk.Label(master=root, label='tkInter back in town')

   # Positioning
   quit_button.pack()
   txHell.pack()
   image_label.pack()
   print 'GUI items initialized...'

   # Key binding

   return image_label

def HOTKEY_setup(root,p):
   # root.bind( '<Escape>', quit_(root, p) )
   pass


if __name__ == '__main__':
   queue = Queue()
   print 'queue initialized...'
   root = tk.Tk()
   image_label = GUI_setup(root)

   p = Process(target=image_capture, args=(queue,))

   HOTKEY_setup(root,p)

   p.start()
   print 'image capture process has started...'


   # setup the update callback
   root.after(0, func=lambda: update_all(root, image_label, queue))
   print 'root.after was called...'
   root.mainloop()
   print 'mainloop exit'
   p.join()
   print 'image capture process exit'
