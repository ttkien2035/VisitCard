from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import os
#import pytesseract
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from PIL import Image

from keras.models import load_model
import keras
import itertools
import string

#edit this line
#pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
#edit this line
path = '161.jpg'
image = cv2.imread(path)
orig = image.copy()
(H, W) = image.shape[:2]
(newW, newH) = (320, 320)
rW = W / float(newW)
rH = H / float(newH)
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('frozen_east_text_detection.pb')
# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
print("Load model")
model=load_model('model_weights.h5')
print("Load success")
results=[]
# loop over the number of rows
def predictions(prob_score, geo):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []
	# loop over rows
	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		# loop over the number of columns
		for i in range(0, numC):
			if scoresData[i] < 0.5:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			# extracting the rotation angle for the prediction and computing the sine and cosine
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# using the geo volume to get the dimensions of the bounding box
			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			# compute start and end for the text pred bbox
			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])
	# return bounding boxes and associated confidence_val
	return (boxes, confidence_val)
def words_from_labels(labels):
   letters= '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
   txt=[]
   for ele in labels:
       if ele == len(letters): # CTC blank space
           txt.append("")
       else:
           #print(letters[ele])
           txt.append(letters[ele])
   return "".join(txt)

def decode_label(out):
   """
   Takes the predicted ouput matrix from the Model and returns the output text for the image
   """
   out_best = list(np.argmax(out[0,2:], axis=1))

   out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value

   outstr=words_from_labels(out_best)
   return outstr
  
def test_data_single_image_Prediction(test_img_path,startX, startY, endX, endY):
   """
   Takes the best model, test data image paths, test data groud truth labels and pre-processes the input image to 
   appropriate format for the model prediction, takes the predicted output matrix and uses best path decoding to 
   generate predicted text and prints the Predicted Text Label, Time Taken for Computation
   """
    
   test_img=cv2.imread(test_img_path)
   # face = image[startY:endY, startX:endX]
   test_img = test_img[startY:endY, startX:endX]
   test_img_resized=cv2.resize(test_img,(170,32))
   test_image=test_img_resized[:,:,1]
   test_image=test_image.T 
   test_image=np.expand_dims(test_image,axis=-1)
   test_image=np.expand_dims(test_image, axis=0)
   test_image=test_image/255
   model_output=model.predict(test_image)
   return model_output
def model_predict(img_path,startX, startY, endX, endY):
   '''
       helper method to process an uploaded image
   '''
   preds=test_data_single_image_Prediction(img_path,startX, startY, endX, endY)
   return preds
(boxes, confidence_val) = predictions(scores, geometry)
boxes = non_max_suppression(np.array(boxes), probs=confidence_val)

results = []
mytext=[]
rois=[]
# loop over the bounding boxes to find the coordinate of bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)
	#extract the region of interest
	r = orig[startY:endY, startX:endX]
	rois.append(r)
	##This will recognize the text from the image of bounding box
	preds = model_predict(path,startX, startY, endX, endY)
	predicted_output=decode_label(preds)
	text = predicted_output

	#append bbox coordinate and associated text to the list of results 
	results.append((startX, startY, endX, endY))
	mytext.append(text)



for text,(start_X, start_Y, end_X, end_Y) in zip(mytext,results):
	pic=orig.copy()
	pos=(start_X, start_Y, end_X, end_Y)
	print(pos)
	print("{}\n".format(text))
	cv2.rectangle(pic, (start_X, start_Y), (end_X, end_Y),
   (0, 0, 0), 2)
	
orig_image=orig.copy()
#Moving over the results and display on the image
for text,(start_X, start_Y, end_X, end_Y) in zip(mytext,results):
   # display the text detected 
   print("{}\n".format(text))
   # Displaying text
   text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
   cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
   (0, 0, 0), 2)

cv2.imshow('abcd',orig_image)
cv2.waitKey()