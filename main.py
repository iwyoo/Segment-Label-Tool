#-------------------------------------------------------------------------------
# Base code reference: https://github.com/puzzledqs/BBox-Label-Tool
# Description: A simple labeling tool for segmentation
# Author: Inwan Yoo (iwyoo@lunit.io)
#-------------------------------------------------------------------------------

from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import os
import glob

import cv2
import numpy as np
from skimage.segmentation import find_boundaries

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

def RGB2HEX(rgb):
    # Ref: https://gist.github.com/mezklador/aadbd8e63d1a4ac207ebc3b45fd8d082
    return "#" + "".join([format(val, '02X') for val in rgb])

MIN_CURSOR_SIZE = 1
UNKNOWN = 255
MAX_CLASS = len(COLORS)
NUM_DRAW_TICK = 10

class LabelTool():
    def __init__(self, master):
        # set up the main frame
        self.parent = master
        self.parent.title("LabelTool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=FALSE, height=FALSE)

        # initialize global state
        self.imageDir = None
        self.labelDir = None
        self.imageList = []
        self.outDir = ''
        self.cur = 0
        self.cur_cls = 0
        self.total = 0
        self.category = 0
        self.labelpath = ''
        self.tkimg = None

        self.click_pos = False
        self.ready = False
        self.radius = 3

        self.draw_tick = 0

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.imageLabel = Label(self.frame, text="Image Dir:")
        self.imageLabel.grid(row=0, column=0, sticky=E)
        self.imagePath = Label(self.frame)
        self.imagePath.grid(row=0, column=1, sticky=W+E)
        self.labelLabel = Label(self.frame, text="Label Dir:")
        self.labelLabel.grid(row=1, column=0, sticky=E)
        self.labelPath = Label(self.frame)
        self.labelPath.grid(row=1, column=1, sticky=W+E)
        self.loadImageBtn = Button(self.frame, text="Load image directory", 
                                   command=self.loadImageDir)
        self.loadImageBtn.grid(row=0, column=2, sticky=W+E)
        self.loadlabelBtn = Button(self.frame, text="Load label directory", 
                                   command=self.loadLabelDir)
        self.loadlabelBtn.grid(row=1, column=2, sticky=W+E)


        # main panel for labeling
        self.mainPanel = Canvas(self.frame)
        self.mainPanel.bind("<Button-1>", self.mouseClickPos)
        self.mainPanel.bind("<Button-3>", self.mouseClickNeg)
        self.mainPanel.bind("<B1-Motion>", self.mouseMovePos)
        self.mainPanel.bind("<B3-Motion>", self.mouseMoveNeg)
        self.mainPanel.bind("<ButtonRelease-1>", self.drawImage)
        self.mainPanel.bind("<ButtonRelease-3>", self.drawImage)
        self.mainPanel.grid(row=2, column=1, rowspan=3, sticky=W+N)

        self.parent.bind("a", self.prevImage)
        self.parent.bind("d", self.nextImage)
        self.parent.bind("q", self.prevClass)
        self.parent.bind("e", self.nextClass)
        self.parent.bind("w", self.cursorDilate)
        self.parent.bind("s", self.cursorErode)
        self.parent.bind("x", self.saveImage)

        self.classLabel = Label(self.frame)
        self.classLabel.grid(row=2, column=2, sticky=W+N)
        self.classLabel.config(text="Class select: {}".format(self.cur_cls), 
                               bg=RGB2HEX(COLORS[self.cur_cls]))

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

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)

    def loadLabelDir(self):
        self.labelDir = fd.askdirectory(initialdir=".")
        self.parent.focus()
        assert self.labelDir != self.imageDir

        # set up output dir
        os.makedirs(self.labelDir, exist_ok=True)

        self.labelPath.config(text=self.labelDir)

        if self.imageDir and self.labelDir:
            self.loadImage()
            print("{:d} images loaded from {:s}".format(self.total, self.imageDir))

            self.ready = True

    def loadImageDir(self):
        self.imageDir = fd.askdirectory(initialdir=".")
        self.parent.focus()
        assert self.imageDir != self.labelDir

        # get image path list
        self.imageList = [] 
        for ext in [".jpg", ".JPEG", ".png", ".PNG"]:
            self.imageList.extend(
                glob.glob(os.path.join(self.imageDir, '*{}'.format(ext))))

        if len(self.imageList) == 0:
            print("No '.jpg', '.JPEG', '.png', '.PNG' images found in the"
                  "specified dir!")
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

        self.imagePath.config(text=self.imageDir)

        if self.imageDir and self.labelDir:
            self.loadImage()
            print("{:d} images loaded from {:s}".format(self.total, self.imageDir))

            self.ready = True

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
            self.label_arr = np.full(self.image_arr.shape[:2], UNKNOWN, dtype=np.uint8)
        self.drawImage()

    def drawImage(self, event=False):
        # Draw color on image
        known = self.label_arr != 255
        self.color_arr[known] = COLORS[self.label_arr[known]] * 0.6 \
                                + self.image_arr[known] * 0.4
        self.color_arr[~known] = self.image_arr[~known]

        # Draw edge
        edge = find_boundaries(self.label_arr)
        self.color_arr[edge, :] = 0
        
        # Set tkimg
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(self.color_arr))
        self.mainPanel.config(width=max(self.tkimg.width(), 400),
                              height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)

        # Write image
        self.progLabel.config(text="%04d/%04d" %(self.cur, self.total))

    def saveImage(self, event=None):
        with open(self.labelpath, 'w') as f:
            cv2.imwrite(self.labelpath, self.label_arr)
        print("Image No. {:d} saved".format(self.cur))

    def mouseClickPos(self, event, draw_with_tick=False):
        self.click_pos = True
        if self.ready:
            x, y = event.x, event.y
            cv2.circle(self.label_arr, (x, y), self.radius, self.cur_cls,
                       thickness=-1)
            if draw_with_tick:
                self.draw_tick = (self.draw_tick + 1) % NUM_DRAW_TICK
                if self.draw_tick == 0:
                    self.drawImage()
            else:
                self.drawImage()
        self.click_pos = False

    def mouseClickNeg(self, event, draw_with_tick=False):
        if self.ready:
            x, y = event.x, event.y
            cv2.circle(self.label_arr, (x, y), self.radius, UNKNOWN,
                       thickness=-1)
            if draw_with_tick:
                self.draw_tick = (self.draw_tick + 1) % NUM_DRAW_TICK
                if self.draw_tick == 0:
                    self.drawImage()
            else:
                self.drawImage()

    def mouseMovePos(self, event=None):
        self.mouseClickPos(event, draw_with_tick=True)

    def mouseMoveNeg(self, event=None):
        self.mouseClickNeg(event, draw_with_tick=True)

    def cursorDilate(self, event=None):
        self.radius += 1

    def cursorErode(self, event=None):
        if self.radius > 1:
            self.radius -= 1

    def prevClass(self, event=None):
        if self.cur_cls > 0:
            self.cur_cls -= 1
            self.classLabel.config(
                text="Class select: {}".format(self.cur_cls), 
                bg=RGB2HEX(COLORS[self.cur_cls]))

    def nextClass(self, event=None):
        if self.cur_cls < MAX_CLASS - 1:
            self.cur_cls += 1
            self.classLabel.config(
                text="Class select: {}".format(self.cur_cls), 
                bg=RGB2HEX(COLORS[self.cur_cls]))

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
