#-------------------------------------------------------------------------------
# Base code reference: https://github.com/puzzledqs/BBox-Label-Tool
# Description: A simple labeling tool for segmentation
# Author: Inwan Yoo (iwyoo@lunit.io)
#-------------------------------------------------------------------------------

from Tkinter import *
import tkMessageBox
from PIL import Image, ImageTk
import os
import glob

import cv2
import numpy as np
from shapely.geometry import Point

# colors for the segmentation 
# color-map of mapillary_vistas dataset
# Ref: https://github.com/tensorflow/models/blob/master/research/deeplab/utils/get_dataset_colormap.py#L247
COLORS = np.asarray([
    [165, 42, 42],
    [0, 192, 0],
    [196, 196, 196],
    [190, 153, 153],
    [180, 165, 180],
    [102, 102, 156],
    [102, 102, 156],
    [128, 64, 255],
    [140, 140, 200],
    [170, 170, 170],
    [250, 170, 160],
    [96, 96, 96],
    [230, 150, 140],
    [128, 64, 128],
    [110, 110, 110],
    [244, 35, 232],
    [150, 100, 100],
    [70, 70, 70],
    [150, 120, 90],
    [220, 20, 60],
    [255, 0, 0],
    [255, 0, 0],
    [255, 0, 0],
    [200, 128, 128],
    [255, 255, 255],
    [64, 170, 64],
    [128, 64, 64],
    [70, 130, 180],
    [255, 255, 255],
    [152, 251, 152],
    [107, 142, 35],
    [0, 170, 30],
    [255, 255, 128],
    [250, 0, 30],
    [0, 0, 0],
    [220, 220, 220],
    [170, 170, 170],
    [222, 40, 40],
    [100, 170, 30],
    [40, 40, 40],
    [33, 33, 33],
    [170, 170, 170],
    [0, 0, 142],
    [170, 170, 170],
    [210, 170, 100],
    [153, 153, 153],
    [128, 128, 128],
    [0, 0, 142],
    [250, 170, 30],
    [192, 192, 192],
    [220, 220, 0],
    [180, 165, 180],
    [119, 11, 32],
    [0, 0, 142],
    [0, 60, 100],
    [0, 0, 142],
    [0, 0, 90],
    [0, 0, 230],
    [0, 80, 100],
    [128, 64, 64],
    [0, 0, 110],
    [0, 0, 70],
    [0, 0, 192],
    [32, 32, 32],
    [0, 0, 0],
    [0, 0, 0],
])

MIN_CURSOR_SIZE = 1
UNKNOWN = 255

class LabelTool():
    def __init__(self, master):
        # set up the main frame
        self.parent = master
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=FALSE, height=FALSE)

        # initialize global state
        self.imageDir = ''
        self.labelDir = ''
        self.imageList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.labelpath = ''
        self.tkimg = None

        self.radius = 3
        self.class_idx = 0


        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.image_text = Label(self.frame, text="Image Dir:")
        self.image_text.grid(row=0, column=0, sticky=E)
        self.image_entry = Entry(self.frame)
        self.image_entry.grid(row=0, column=1, sticky=W+E)
        self.label_text = Label(self.frame, text="Label Dir:")
        self.label_text.grid(row=1, column=0, sticky=E)
        self.label_entry = Entry(self.frame)
        self.label_entry.grid(row=1, column=1, sticky=W+E)
        self.loadBtn = Button(self.frame, text="Load", command=self.loadDir)
        self.loadBtn.grid(row=0, column=2, sticky=W+E)


        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='circle') # CHECK: circle?
        self.mainPanel.bind("<Button-1>", self.mouseClickPos)
        self.mainPanel.bind("<Button-2>", self.mouseClickNeg)
        self.mainPanel.bind("<Motion>", self.mouseMove) # TODO
        self.mainPanel.grid(row=2, column=1, rowspan=3, sticky=W+N)

        self.parent.bind("a", self.prevImage)
        self.parent.bind("d", self.nextImage)
        self.parent.bind("q", self.prevClass)
        self.parent.bind("e", self.nextClass)
        self.parent.bind("w", self.cursorDilate)
        self.parent.bind("s", self.cursorErode)
        self.parent.bind("x", self.saveImage)

        # showing bbox info & delete bbox
        self.class_lbl = Label(self.frame, text='Class select:')
        self.class_lbl.grid(row=1, column=2, sticky=W+N)
        self.listbox = Listbox(self.frame, width=22, height=12)
        self.listbox.grid(row=2, column=2, sticky=N) # CHECK: sticky W+N+E?

        self.btnAdd = Button(self.frame, text='Add', command=self.addClass)
        self.btnAdd.grid(row=3, column=2, sticky=W+E+N)
        self.btnDel = Button(self.frame, text='Delete', command=self.delClass)
        self.btnDel.grid(row=4, column=2, sticky=W+E+N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row=5, column=1, columnspan=2, sticky=W+E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevImage)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.nextImage)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="Go to Image No.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImage)
        self.goBtn.pack(side=LEFT)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)

        self.frame.columnconfigure(1, weight=1) # CHECK?
        self.frame.rowconfigure(4, weight=1)    # CHECK?

    def loadDir(self):
        # get image directory from self.entry
        self.imageDir = self.image_entry.get()
        self.labelDir = self.label_entry.get()
        self.parent.focus()  # CHECK?

        # get image path list
        self.imageList = [] 
        for ext in [".jpg", ".JPEG", ".png", ".PNG"]:
            self.imageList.extend(
                glob.glob(os.path.join(self.imageDir, '*{}'.format(ext)))

        if len(self.imageList) == 0:
            print("No '.jpg', '.JPEG', '.png', '.PNG' images found in the"
                  "specified dir!")
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

        # set up output dir
        os.makedirs(self.label_dir, exist_ok=True)

        self.loadImage()
        print("{:d} images loaded from {:s}".format(self.total, self.imageDir))

    def loadImage(self):
        # load image & label
        imagepath = self.imageList[self.cur-1]
        labelname = os.path.splitext(os.path.basename(imagepath))[0] + '.png'
        self.labelpath = os.path.join(self.labelDir, labelname)

        self.image_arr = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2RGB)
        self.color_arr = np.zeros_like(self.image_arr)

        if os.path.exists(self.labelpath):
            self.label_arr = cv2.imread(self.labelpath, 0)
        else:
            self.label_arr = np.full(image_arr.shape[:2], UNKNOWN, dtype=np.uint8)
        self.drawImage()

    def drawImage(self):
        known = self.label_arr != 255
        self.color_arr[known] = COLORS[self.label_arr[known]] * 0.6 \
                                + self.image_arr[known] * 0.4
        self.color_arr[~known] = self.image_arr[~known]
        
        # TODO: tk
        # FUTURE: Edge


    def saveImage(self):
        with open(self.labelpath, 'w') as f:
            cv2.imwrite(self.labelpath, self.label_arr)
        print("Image No. {:d} saved".format(self.cur))

    def mouseClickPos(self, event):
        x, y = event.x, event.y
        cv2.circle(self.label_arr, (x, y), radius=self.radius, self.class_idx,
                   thickness=-1)
        self.drawImage()

    def mouseClickNeg(self, event):
        x, y = event.x, event.y
        cv2.circle(self.label_arr, (x, y), radius=self.radius, UNKNOWN,
                   thickness=-1)
        self.drawImage()

    def cursorDilate(self, event=None):
        self.radius += 1

    def cursorErode(self, event=None):
        if self.radius > 1:
            self.radius -= 1

    def mouseMove(self, event):
        # TODO
        pass

    def addClass(self, event=None):
        # TODO
        pass

    def delClass(self, event=None):
        # TODO
        pass

    def prevClass(self, event=None):
        # TODO
        pass

    def nextClass(self, event=None):
        # TODO
        pass

    def prevImage(self, event=None):
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextImage(self, event=None):
        self.saveImage()
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()

if __name__ == '__main__':
    root = Tk()
    tool = LabelTool(root)
    root.resizable(width=True, height=True)
    root.mainloop()
