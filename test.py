
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
#from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
#from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
#from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

path = "/truba_scratch/agasi/test"
img_path = "/truba_scratch/agasi/test/images"

INIT_LR = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 32

print("[INFO] loading dataset...")
data = []
labels = []
s_bbox = []
o_bbox = []
imagePaths = []

for csvPath in paths.list_files(path, validExts=(".csv")):
	rows = open(csvPath).read().strip().split("\n")

	for row in rows:
		row = row.split(";")
		(obj, subject, predicate, subx, suby, subw, subh, objx, objy, objw, objh, image_id) = row
		#(_, _, predicate, subx, suby, subw, subh, objx, objy, objw, objh, _) = row
		
		imgPath = os.path.join(img_path,image_id+".jpg")
		image = cv2.imread(imgPath)
		(h, w) = image.shape[:2]
	
		subx = float(subx) / w
		suby = float(suby) / h
		subw = float(subw) / w
		subh = float(subh) / h

		objx = float(objx) / w
		objy = float(objy) / h
		objw = float(objw) / w
		objh = float(objh) / h

		image = load_img(imgPath, target_size=(224, 224))
		image = img_to_array(image)

		data = np.append(data, image)
		labels = np.append(labels, predicate)
		s_bbox = np.append(s_bbox, (subx, suby, subw, subh))
		o_bbox = np.append(o_bbox, (objx, objy, objw, objh))
		imagePaths.append(imgPath)

data = np.array(data, dtype="float32") / 255.0
s_bbox = np.array(s_bbox, dtype="float32")
o_bbox = np.array(o_bbox, dtype="float32")
labels = np.array(labels)
imagePaths = np.array(imagePaths) 

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

data = np.reshape(data, (-1,224,224,3))
s_bbox = np.reshape(s_bbox, (-1,1,4))
o_bbox = np.reshape(o_bbox, (-1,1,4))

split_data = train_test_split(data,test_size=0.20, random_state=42)
(trainImages, testImages) = split_data[:2]

split_s_bbox = train_test_split(s_bbox,test_size=0.20, random_state=42)
(trainsBBoxes, testsBBoxes) = split_s_bbox[:2]

split_o_bbox = train_test_split(o_bbox,test_size=0.20, random_state=42)
(trainoBBoxes, testoBBoxes) = split_o_bbox[:2]

split_labels = train_test_split(labels,test_size=0.20, random_state=42)
(trainLabels, testLabels) = split_labels[:2]

split_imgPaths = train_test_split(imagePaths,test_size=0.20, random_state=42)
(trainPaths, testPaths) = split_imgPaths[:2]

print("[INFO] saving testing image paths...")
f = open("/truba_scratch/agasi/test/testPaths.txt", "w")
f.write("\n".join(map(str,testPaths)))
f.close()

mirrored_strategy = tf.distribute.MirroredStrategy()
GLOBAL_BATCH_SIZE = BATCH_SIZE * mirrored_strategy.num_replicas_in_sync

with mirrored_strategy.scope():
	vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
	vgg.trainable = False

	flatten = vgg.output
	flatten = Flatten()(flatten)

	sbboxHead = Dense(128, activation="relu")(flatten)
	sbboxHead = Dense(64, activation="relu")(sbboxHead)
	sbboxHead = Dense(32, activation="relu")(sbboxHead)
	sbboxHead = Dense(4, activation="sigmoid", name="subject_bbox")(sbboxHead)

	obboxHead = Dense(128, activation="relu")(flatten)
	obboxHead = Dense(64, activation="relu")(obboxHead)
	obboxHead = Dense(32, activation="relu")(obboxHead)
	obboxHead = Dense(4, activation="sigmoid", name="object_bbox")(obboxHead)

	predicate = Dense(512, activation="relu")(flatten)
	predicate = Dropout(0.5)(predicate)
	predicate = Dense(512, activation="relu")(predicate)
	predicate = Dropout(0.5)(predicate)
	predicate = Dense(len(lb.classes_), activation="softmax", name="predicate")(predicate)

	model = Model(inputs=vgg.input, outputs=(sbboxHead, obboxHead, predicate))

	losses = { "predicate": "categorical_crossentropy", "subject_bbox": "mean_squared_error", "object_bbox": "mean_squared_error", }
	lossWeights = { "predicate": 1.0, "subject_bbox": 1.0, "object_bbox": 1.0 }

	opt = Adam(learning_rate=1e-4)
	model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)

print(model.summary())

trainTargets = {"subject_bbox": trainsBBoxes, "object_bbox": trainoBBoxes, "predicate": trainLabels}
testTargets = {"subject_bbox": testsBBoxes, "object_bbox": testoBBoxes, "predicate": testLabels}

print("[INFO] training model...")
H = model.fit(trainImages, trainTargets, validation_data=(testImages, testTargets),batch_size=GLOBAL_BATCH_SIZE,epochs=NUM_EPOCHS,verbose=1)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save("/truba_scratch/agasi/test/detector.h5", save_format="h5")

# serialize the label binarizer to disk
print("[INFO] saving label binarizer...")
f = open("/truba_scratch/agasi/test/lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the total loss, label loss, and bounding box loss
lossNames = ["loss", "subject_bbox_loss", "object_bbox_loss", "predicate_loss"]
N = np.arange(0,NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(4, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch")
	ax[i].set_ylabel("Loss")
	ax[i].plot(N, H.history[l], label=l)
	ax[i].plot(N, H.history["val_" + l], label="val_" + l)
	ax[i].legend()

#plt.show()
plt.tight_layout()
plotPath = os.path.sep.join([path, "losses.png"])
plt.savefig(plotPath)
plt.close()


plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["predicate_accuracy"],label="predicate_accuracy_acc")
plt.plot(N, H.history["val_predicate_accuracy"],label="val_predicate_accuracy_acc")
plt.title("Predicate Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
#plt.show()

plotPath = os.path.sep.join([path, "accs.png"])
plt.savefig(plotPath)

print("[INFO] loading object detector...")
model = load_model("/truba_scratch/agasi/test/detector.h5")
lb = pickle.loads(open("/truba_scratch/agasi/test/lb.pickle", "rb").read())

imagePaths = open("/truba_scratch/agasi/test/testPaths.txt").read().strip().split("\n")

print(lb.classes_)

width=800
height=600
dim = (width, height)

for imagePath in imagePaths:
  image = load_img(imagePath, target_size=(224, 224))
  image = img_to_array(image) / 255.0
  image = np.expand_dims(image, axis=0)

  (s_bboxPreds, o_bboxPreds, prediction) = model.predict(image)
  (s_bbox_startX, s_bbox_startY, s_bbox_endX, s_bbox_endY) = s_bboxPreds[0]
  (o_bbox_startX, o_bbox_startY, o_bbox_endX, o_bbox_endY) = o_bboxPreds[0]
  
  print(imagePath)
  print(s_bboxPreds[0], "\t")
  print(o_bboxPreds[0], "\t")
  i = np.argmax(prediction, axis=1)
  label = lb.classes_[i][0]
  print(i, "\t")
  print(label, "\t")

  image = cv2.imread(imagePath)
  image = cv2.resize(image,dim,interpolation = cv2.INTER_AREA)
  (h, w) = image.shape[:2]

  print(s_bbox_startX, s_bbox_startY, s_bbox_endX, s_bbox_endY)
  s_bbox_startX = int(s_bbox_startX * w)
  s_bbox_startY = int(s_bbox_startY * h)
  s_bbox_endX = int(s_bbox_endX * w)
  s_bbox_endY = int(s_bbox_endY * h)
  print(s_bbox_startX, s_bbox_startY, s_bbox_endX, s_bbox_endY)

  print(o_bbox_startX, o_bbox_startY, o_bbox_endX, o_bbox_endY)
  o_bbox_startX = int(o_bbox_startX * w)
  o_bbox_startY = int(o_bbox_startY * h)
  o_bbox_endX = int(o_bbox_endX * w)
  o_bbox_endY = int(o_bbox_endY * h)
  print(o_bbox_startX, o_bbox_startY, o_bbox_endX, o_bbox_endY)

  #cv2.putText(image, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  #cv2.rectangle(image, (s_bbox_startX,s_bbox_startY), (s_bbox_endX,s_bbox_endY), (255, 0, 0), 2)
  #cv2.rectangle(image, (o_bbox_startX,o_bbox_startY), (o_bbox_endX,o_bbox_endY), (0, 0, 255), 2)
  #cv2_imshow(image[:, :, ::-1])

