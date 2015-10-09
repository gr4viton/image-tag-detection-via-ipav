#!/usr/bin/python

# http://stackoverflow.com/questions/17073227/display-an-opencv-video-in-tkinter-using-multiprocessing
import array

import numpy as np
from multiprocessing import Process, Queue
from Queue import Empty
import cv2
# import cv2.cv as cv
from PIL import Image, ImageTk
import time
import Tkinter as tk

from thisCV import *


# import sys

# tkinter GUI functions----------------------------------------------------------
global maxLenQueue

def quit_(root, process, *whatever):
    process.terminate()
    root.destroy()


# def quitCallback():

def update_image(image_label, frame):
    if len(frame) == 0:
        return
    if len(frame.shape) == 2:
        im = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        print frame.shape
        print len(frame)
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #im = frame

    a = Image.fromarray(im)
    b = ImageTk.PhotoImage(image=a)
    image_label.configure(image=b)
    image_label._image_cache = b  # avoid garbage collection
    root.update()

def update_image_tag(image_label, imTags):
    numTags = len(imTags)
    if numTags  == 0:
        return
    global strNumTags
    global slTags
    strNumTags.set(numTags )

    k = slTags.get()
    if k >= numTags :
        k = numTags  - 1
    update_image(image_label, imTags[k])

def update_all(root, params):
    imlTags, queTag, imlLabel, queue = params
    update_image_tag(imlTags, queTag.get())


    # # update_image(imLabel, queue)
    # if queue.qsize() < maxLenQueue:
    update_image(imlLabel, queue.get())

    root.after(0, func=lambda: update_all(root, params))


# multiprocessing image processing functions-------------------------------------
def image_capture(queue, queTag):
    global maxLenQueue
    maxLenQueue = 5
    cap = cv2.VideoCapture(0)
    loopingCV = 1
    while loopingCV:
        imWhole, imTags = stepCV(cap)
        if queue.qsize() < maxLenQueue:
            queue.put(imWhole)
        queTag.put(imTags)
    cap.release()


def GUI_setup(root):
    # GUI Items
    # print 'GUI initialized...'
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #:: frAddList
    # ____________________________________________________ frames
    frLeft = tk.LabelFrame(root, padx=5, width= 320, text="Add by list" )
    frTags = tk.LabelFrame(root, padx=5, width= 320, text="Detected tags")
    # ____________________________________________________ images
    imlLabel = tk.Label(frLeft)
    imlTags = tk.Label(frTags)
    # ____________________________________________________ texts
    # ____________________________________________________ sliders
    global slTags
    slTags = tk.Scale(frTags, from_=0, to_=10, orient=tk.HORIZONTAL)
    # ____________________________________________________ entries
    enHell = tk.Entry(frLeft, text='tkInter back in town')
    global strNumTags
    strNumTags = tk.StringVar()
    lbNumTags = tk.Label(frTags, textvariable=strNumTags)
    strNumTags.set( "0  found" )

    # ____________________________________________________ buttons
    btnQuit = tk.Button(frLeft, text='Q', command=lambda: quit_(root, p))
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # GRID of Frames
    #____________________________________________________
    frLeft.grid (row=2, column=2, rowspan=1, columnspan=1, sticky=tk.NSEW)

    imlLabel.grid(row=3, column=2, rowspan=1, columnspan=1, sticky=tk.NSEW )
    enHell.grid (row=1, column=2, rowspan=1, columnspan=1, sticky=tk.NSEW )
    btnQuit.grid(row=0, column=3, rowspan=4, columnspan=1, sticky=tk.NSEW )
    #____________________________________________________
    frTags.grid (row=2, column=3, rowspan=1, columnspan=1, sticky=tk.NSEW )

    lbNumTags.grid(row=1, column=2, rowspan=1, columnspan=1)#, sticky=tk.NSEW )
    slTags.grid(row=2, column=2, rowspan=1, columnspan=1, sticky=tk.NSEW )
    imlTags.grid(row=3, column=2, rowspan=1, columnspan=1, sticky=tk.NSEW )

    print 'GUI initialized...'

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Key binding

    # return
    return imlLabel, imlTags


def HOTKEY_setup(root, p):
    # root.bind( '<Escape>', quit_(root, p) )
    pass

if __name__ == '__main__':
    global maxLenQueue
    maxLenQueue = 1

    queue = Queue( )
    queTag = Queue( )

    print 'queue initialized...'
    root = tk.Tk()
    imlLabel, imlTags = GUI_setup(root)


    p = Process(target=image_capture, args=(queue, queTag,))

    HOTKEY_setup(root, p)

    p.start()
    print 'image capture process has started...'

    root.minsize(width=640, height=100)

    # setup the update callback (recursive calling inside)
    params = imlTags, queTag, imlLabel, queue
    root.after(0, func=lambda: update_all(root, params))


    print 'root.after was called...'
    root.mainloop()
    print 'mainloop exit'
    p.terminate()
    # p.join()
    print 'image capture process exit'
