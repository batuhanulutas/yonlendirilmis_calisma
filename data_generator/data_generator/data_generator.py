# Gerekli kütüphaneleri içe aktarıyoruz.

import os 
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Flatten, Dense , Dropout , Activation
from keras import backend as K 

# Verimizin CNN yapısı.
# Model adında değer döndürüyor. İçerisinde 4 parametre belirtiyoruz.
def build_model(width,height,depth,classes):
    model = Sequential()
    inputShape = (height,width,depth)
    
    if K.image_data_format () == "channels_first":
        inputShape = (depth,height,width)

    model.add(Conv2D(32,kernel_size=(3,3),activation="relu",padding="same",input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


    model.add(Conv2D(64,kernel_size=(3,3),activation="relu",padding="same",input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

# Gerekli veriyi ve modelin ismini nasıl kaydetmesini beliriyoruz.
modelname="data"
path = 'data/'

# Verimizi 50x50 piksellerle eğitiyoruz.
resize= 50

data=[]
labels=[]


EPOCHS=30
INIT_LR=1e-3
BS=32

# XML uzantılı ve görüntülerimize erişiyoruz.
# XML uzantılı dosyalarımızda <object> ve <bndbox> verilerine erişiyoruz.
# img_to_array fonksiyonu ile verilerimizi diziye çeviriyoruz.

for file in os.listdir(path):
    if not file.endswith(".xml"): continue

    full=os.path.join(path,file)

    print(full)
    tree=ET.parse(full)

    root=tree.getroot()

    fileimg=root[1].text
    print(fileimg)
    image=cv2.imread(path+fileimg)

    print(root[1].text)
    objects=root.findall("object")
    for obj in objects:

        name=obj[0].text
        print(name)
        items=obj.findall("bndbox")
        xmin=int(items[0][0].text)
        ymin=int(items[0][1].text)
        xmax=int(items[0][2].text)
        ymax=int(items[0][3].text)
        print(xmin,ymin,xmax,ymax)

        patch = image[ymin:ymax,xmin:xmax,:]
        patch=cv2.resize(patch,(resize,resize))
        
        x=img_to_array(patch)

        data.append(x)

        label=0

        if name == "ball":
            label=1
        labels.append(label)

# Normalize ediyoruz
data = np.array(data,dtype="float")/255.0
labels = np.array(labels)

# Verimizi yüzde 25 olarak test verisi olarak ayırıyoruz
(train_x , test_x , train_y , test_y)=train_test_split(data , labels , test_size = 0.25 , random_state = 42)

# Verimizi kategori haline getiriyoruz. Burada kategori olarak "ball" ve "goal" olarak ayrılmıştır.
train_y = to_categorical(train_y , num_classes = 2)
test_y = to_categorical(test_y , num_classes = 2)

# Veri Üretme
aug=ImageDataGenerator(rotation_range = 30 , width_shift_range = 0.1 , height_shift_range = 0.1 , shear_range = 0.2 , zoom_range = 0.3 , fill_mode = "nearest")


model = build_model(width=resize,height=resize,depth=3,classes=2)
opt= Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

# Modelimize fit işlemi yaparak eğitimi gerçekleştiriyoruz
Hist=model.fit_generator(aug.flow(train_x,train_y,batch_size=BS),validation_data=(test_x,test_y), steps_per_epoch=len(train_x)//BS, epochs=EPOCHS, verbose=1)

# Eğitilen modeli kayıt ediyoruz
model.save(modelname+".model")

# Eğilen modeli çizdiriyoruz.
plt.figure()
plt.plot(Hist.history["loss"],label="Eğitim Loss")
plt.plot(Hist.history["val_loss"],label="Doğrulama Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(Hist.history["accuracy"],label="Eğitim accuracy")
plt.plot(Hist.history["val_accuracy"],label="Doğrulama accuracy")
plt.legend()
plt.show()

