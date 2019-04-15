# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 15:09:43 2019

@author: Kedar Kulkarni
"""

import numpy as np
from matplotlib import pyplot as io
from PIL import Image
import random
import matplotlib.image as mpimg
import os
import sys

path=sys.argv[1]
#path=("C:/Users/kedar/Desktop/Koala.jpg")
base=os.path.dirname(path)
print("Original Image")
img = io.imread(path)
imgplot = io.imshow(img)
io.show()

kl=[2,5,10,15,20]
for k in kl:
  kmap={}
  for i in range(k):
    l,m,n=img.shape
    img1=img.reshape(l*m,n)
    kmap[i]=img1[random.randint(0,l*m),0:n]

  count=0
  while (count<20):
    count+=1
    d = np.empty((l*m,1))
    for i in range(k):
      d1 = np.sqrt(np.linalg.norm(img1.reshape(-1,n)-kmap[i].reshape(-1,n),axis=1)**2).reshape(-1,1)
      d = np.concatenate((d,d1), axis =1)
    d = np.delete(d,[0], axis = 1)
    cluster_centroid = np.argmin(d, axis = 1).reshape(-1,1)

    for i in range(k):
      i1, j1 =  np.where(cluster_centroid==i)
      kmap[i] = np.average(img1[i1], axis = 0)

  img1=np.reshape(img1,(l,m,n))
  add =cluster_centroid.reshape(l,m,1)
  img1=np.dstack((img1,add))
  for r in range(k):
    img1[img1[:,:,n]==r]=[np.append(kmap[r],(r))]

  finalImg=img1[:,:,0:n]
  img2 = np.asarray(finalImg).reshape(l,m,n)
  array = np.zeros([l,m,n], dtype = np.uint8)
  array = img2
  img2 = Image.fromarray(array.astype('uint8'))
  img2.save(base+str(k)+".png")
  print("Image reproduced successfully for k = "+str(k))

  img2=mpimg.imread(base+str(k)+".png")
  imgplot = io.imshow(img2)
  io.show()