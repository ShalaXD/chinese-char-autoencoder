
# coding: utf-8

# In[1]:


from PIL import Image


# In[13]:


import json
from pprint import pprint

with open('char.json') as f:
    data = json.load(f)

pprint(data["gbk"][0:5])


# In[33]:


print(type(data['gbk'][0]))
data["gbk"][0].encode('utf8')


# In[92]:


from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from matplotlib.pyplot import imshow
import numpy as np

get_ipython().magic('matplotlib inline')

fnt = ImageFont.truetype('heiti.ttc', 64)

train_x = None


for i in range(0,3):
    source_img = Image.new("L", (64, 64),(0))    
    draw = ImageDraw.Draw(source_img)
    draw.text((0,0), data["gbk"][i], font=fnt, fill=(255))
#     imshow(np.asarray(source_img), cmap='gray')
#     source_img.save("outfile"+str(i), "png")
    source_img = np.asarray(source_img)
    if train_x is None:
        train_x = np.asarray([source_img])
    else:
        train_x = np.concatenate((train_x, np.asarray([source_img])), axis = 0)


# In[93]:


print(train_x.shape)


# In[ ]:




