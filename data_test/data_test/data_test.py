
# Gerekli kütüphaneleri içe aktarıyoruz.

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2

# Gerekli olan eğitilmiş veriyi ve kullanılacak olan görüntümüzün yollarını belirtiyoruz.
# Verimiz 50x50 eğitilmiştir.

modelname = "data.model"
img_file = "resize.jpg"
resize = 50
    

# Resmimizi içe aktarıyoruz.
image = cv2.imread(img_file)

# Sınırlıyacı kutular çiziyoruz.
bbox = cv2.selectROI(image,False)

image = image[int(bbox[1]):int(bbox[1]+bbox[3]),int(bbox[0]):int(bbox[0]+bbox[2]),:]

# Görüntüyü kopyalıyoruz

selected = image.copy()

# Eğitilen veriyi işlemek için resmimizi ön işleme alıyoruz. 

image = cv2.resize(image, (resize, resize))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Modelimizi yüklüyoruz.

model = load_model(modelname)

# Görüntümüzü negatif ve pozitif olarak sınıflandırıyoruz.
(negative, pozitive) = model.predict(image)[0]

# Etiketleme işlemi yapıyoruz.
label = "Ball" if pozitive > negative else "goal"
proba = pozitive if pozitive > negative else negative

# Yüzdelik oranını gösteriyoruz.
label = "{}: {:.2f}%".format(label, proba * 100)

# Görüntümüzü etiketliyoruz ve çıkan sonucu görüntülüyoruz
output = cv2.resize(selected,dsize=(400,400))
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)


cv2.imshow("çikti", output)
cv2.waitKey(0)
cv2.destroyAllWindows()