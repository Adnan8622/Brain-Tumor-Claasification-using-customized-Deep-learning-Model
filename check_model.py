
from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf

CATEGORIES = ["Normal", "Tumor"] 
model = load_model('Tumor_model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

img1 = cv2.imread('1(2).jpg') ### Input Images
img = cv2.resize(img1,(96,96))
img = np.reshape(img,[1,96,96,3])

classes = model.predict_classes(img)
#for weights
#cls=model.predict(img)
print(classes)
cv2.putText(img1,CATEGORIES[int(classes)] , (10, img1.shape[0] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
#print(CATEGORIES[int(classes)])
cv2.imshow('Result',img1)
cv2.imwrite('res1.jpg',img1) ### u can save the results
cv2.waitKey(0)

