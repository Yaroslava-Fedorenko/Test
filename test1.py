#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("/Users/yaroslava/downloads/airbus-ship-detection"))


# In[2]:


train = os.listdir('/Users/yaroslava/downloads/airbus-ship-detection/train_v2')
print(len(train))

test = os.listdir('/Users/yaroslava/downloads/airbus-ship-detection/test_v2')
print(len(test))


# In[3]:


submission = pd.read_csv('/Users/yaroslava/downloads/airbus-ship-detection/sample_submission_v2.csv')
submission.head()


# In[4]:


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T 


# In[5]:


masks = pd.read_csv('/Users/yaroslava/downloads/airbus-ship-detection/train_ship_segmentations_v2.csv')
masks.head()


# In[6]:


ImageId = '0a1a7f395.jpg'

img = cv2.imread('/Users/yaroslava/downloads/airbus-ship-detection/train_v2/' + ImageId)
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

# Take the individual ship masks and create a single mask array for all ships
all_masks = np.zeros((768, 768))
for mask in img_masks:
    all_masks += rle_decode(mask)
    

fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()

