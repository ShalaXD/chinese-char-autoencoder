
# coding: utf-8

# In[83]:


import json
from PIL import Image
from pprint import pprint
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from matplotlib.pyplot import imshow
import numpy as np


import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms 
from torchvision.datasets import MNIST
from torchvision.utils import save_image



# In[84]:


with open('char.json') as f:
    char_list = json.load(f)

pprint(char_list["gbk"][0:5])


# In[85]:


get_ipython().run_line_magic('matplotlib', 'inline')
font_size = 28
fnt = ImageFont.truetype('heiti.ttc', 28)
train_x = None
for word in char_list["gbk"]:
    source_img = Image.new("L", (28, 28),(0))    
    draw = ImageDraw.Draw(source_img)
    draw.text((0,0), word, font=fnt, fill=(255))
#     imshow(np.asarray(source_img), cmap='gray')
#     source_img.save("outfile"+str(i), "png")
    source_img = np.asarray(source_img)
    if train_x is None:
        train_x = np.asarray([source_img])
    else:
        train_x = np.concatenate((train_x, np.asarray([source_img])), axis = 0)


# In[92]:


imshow(np.asarray(source_img), cmap='gray')


# In[86]:


# saving and loading data
np.save("train_char_28",train_x)


# In[125]:



train_x = np.load("train_char.npy")
print(train_x.shape)
train_x = torch.from_numpy(train_x)
train_x = train_x/255
train_x = torch.unsqueeze(train_x, 1)
train_x = train_x.type(torch.FloatTensor)
print(train_x.shape)
# all_images = torch.from_numpy(train_x.reshape((26635,1,28,28)))
dataloader = DataLoader(train_x, batch_size=batch_size, shuffle=True)



# In[126]:



if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')

char_dim = 64

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, char_dim, char_dim)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])


# In[127]:


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(char_dim * char_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12))
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, char_dim * char_dim), 
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img = data
        img = img.view(img.size(0), -1)
        img = img.type(torch.FloatTensor)
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')


# In[157]:



import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, char_dim, char_dim)
    return x


num_epochs = 100
batch_size = 32
learning_rate = 1e-3

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# dataset = MNIST('./data', transform=img_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
#         img,_ = data
        img = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data[0]))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './conv_autoencoder.pth')


# In[119]:


print(img.shape)


# In[133]:


imshow(np.asarray(output[10].cpu().view(char_dim, char_dim).detach().numpy()), cmap='gray')


# In[155]:


imshow(np.asarray(train_x[1].view(char_dim, char_dim).detach().numpy()), cmap='gray')


# In[149]:


train_x[0][0].cuda()


# In[151]:


output[0].shape


# In[153]:


num_0 = model.encoder(train_x[0].reshape(char_dim*char_dim).cuda())
num_1 = model.encoder(train_x[1].reshape(char_dim**2).cuda())
changes = generate_latent_space_vectors(num_0, num_1, 10)
pic = to_img(model.decoder(changes).cpu().data)

save_image(pic, '3-9.png')





# In[134]:


num_9 = model.encoder(output[3])


# In[135]:


num_3 = model.encoder(output[0])


# In[136]:


def generate_latent_space_vectors(a, b, steps):
    results = [a * x / steps + b * (steps - x) / steps for x in range(0, steps + 1)]
    return torch.stack(results)


# In[137]:


changes = generate_latent_space_vectors(num_9, num_3, 10)


# In[138]:


pic = to_img(model.decoder(changes).cpu().data)

save_image(pic, '3-9.png')




# In[139]:


x = 0.35
num_93 = model.decoder(num_3 * x + num_9 * (1 - x))
imshow(np.asarray(num_93.cpu().view(28,28).detach().numpy()), cmap='gray')


# In[ ]:





# In[46]:


i = 0
for data in dataloader:
    if i == 0 :
        print(data[0].shape)
    else :
        break


# In[73]:


img = torch.from_numpy(train_x.reshape((26635,1,64,64)))
img = img.view(img.size(0), -1)
img = Variable(img).cuda()
print(img.shape)


# In[79]:


dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# img, _ = data
# img = img.view(img.size(0), -1)
# img = Variable(img).cuda()

print(dataloader.shape)

