import numpy
import matplotlib.image as mpimg
import sys
import re
import shutil
import os
import math

raw_file_dir = './dataset/train2/lightfield/'
spl_file_dir = './dataset/train2/split_matrix/'

if os.path.exists(spl_file_dir):
    shutil.rmtree(spl_file_dir)
os.mkdir(spl_file_dir)

for files in os.listdir(raw_file_dir):
    img = mpimg.imread(raw_file_dir+files)
    [x,y,z] = numpy.shape(img)
    break

numfrm = len(os.listdir(raw_file_dir))
width = y
subwidth = int(y/8)
height = x
subheight = int(x/8)
channel = z

Frame_id = 0
for files in os.listdir(raw_file_dir):
    img = mpimg.imread(raw_file_dir+files)
    sub_dir = spl_file_dir+'/frame_'+str(Frame_id)+'/'
    os.mkdir(sub_dir)
    for row in range(8):
        for col in range(8):
            file_name = str(row)+'_'+str(col)+'.jpg'
            subaperture = numpy.zeros((subheight,subwidth,channel))
            subaperture = img[row*subheight:row*subheight+subheight-1,col*subwidth:col*subwidth+subwidth-1,:]
            mpimg.imsave(sub_dir+file_name,subaperture)
    Frame_id = Frame_id + 1

