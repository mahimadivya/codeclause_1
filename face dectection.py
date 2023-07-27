#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[2]:


import cv2


# In[5]:


image_path = "sam.jpeg"
image = cv2.imread(image_path)


# In[6]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[7]:


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# In[8]:


faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


# In[9]:


for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


# In[11]:


cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




