# USAGE
# python3 liveness_demo.py --model liveness.model --le le.pickle --detector face_detector

# import the necessary packages
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import face_recognition #dlib's face recognition library 
from boltiot import Bolt
import conf, time #import device api key and ids along with telegram details
import emoji, telegram
from PIL import Image #for the frame(numpy array) to be coverted to PIL image object for sending media to telegram
from telegram.ext import Updater
from telegram.ext import CommandHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup


# construct the argument parse and parse the arguments for input of the trained nueral network, face detector and confidence threshold
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#insert global counter and flag variables 
globflag=0 #to send alert after a few frames were detected having unauthorized access
process=True #to skip frames to ease computation on machine for faster face recognition

#if liveness detection declares that the faces are not spoofed, proceed with face recognition
def face_rec(frame):
	#you need to tell python that these variables are global
	global globflag
	global process
	# Load a sample picture and learn how to recognize it.
	sarvagya_image = face_recognition.load_image_file("sarvagya.jpeg")
	face_encoding = face_recognition.face_encodings(sarvagya_image)[0]
	
	# Create arrays of known face encodings and their names
	known_face_encodings = [
		face_encoding
	]
	known_face_names = [
		"Sarvagya"
	]

	# Initialize some variables
	face_locations = []
	face_encodings = []
	face_names = []    
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) #resize the frame for faster computation on it

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]

	# Only process every other frame of video to save time
	#Find all the faces and face encodings in the current frame of video

	if process:    
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		face_names = []

		for face_encoding in face_encodings:
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			name = "Unknown"


			# Using the known face with the smallest distance to the new face
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = known_face_names[best_match_index]
			if name=="Unknown": #if face not recognized, increase counter by 1
				globflag+=1


			face_names.append(name) #appending whether known or unknown face

	process= not process

	#if 40 or more frames were found to have unknown face, send alert by calling function
	if globflag>40: 
		alert(frame)
		print("Intruder detected. Sending alert to Owner!!")
		globflag=0 #reset to 0
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		# Draw a box around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# Draw a label with a name below the face
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
#end face_rec


#this function is to send alert if spoofed face/unknown face is detected in the video stream
def alert(frame):

	#we convert numpy array frame to an RGB image and save as my.png because only images are sent via the telegram bot
	img = Image.fromarray(frame, 'RGB')
	img.save('my.png')
	bot = telegram.Bot(token=conf.TELEGRAM_BOT_ID) #configure telegram chat bot
	
	#configuring and switching on the buzzer and switching off after a certain time
	mybolt = Bolt(conf.API_KEY, conf.DEVICE_ID)
	response = mybolt.digitalWrite('0', 'HIGH')
	#time.sleep(10)
	response = mybolt.digitalWrite('0', 'LOW')

	#We send the captured photo in binary mode of the intruder along with a caption to telegram
	try:
		bot.send_photo(chat_id=conf.TELEGRAM_CHAT_ID, photo=open('my.png', 'rb'), caption="Intruder Alert!") 
	except Exception as e: #raise exception if not sent
		print("An error occurred in sending the alert message via Telegram")
		print(e)
#end alert

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#intialize counter variable
counter=0

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the detected bounding box does fall outside the
			# dimensions of the frame
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

			# extract the face ROI and then preproces it in the exact
			# same manner as our training data
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (32, 32))
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			# pass the face ROI through the trained liveness detector
			# model to determine if the face is "real" or "fake"
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]
			
			#if label is real, do face recognition on the frame
			if label=="real":
				face_rec(frame)
			
			# draw the label and bounding box on the frame
			label0 = "{}".format(label)
			cv2.putText(frame, label0, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)

			#if a spoofed frame is detected increase counter by 1
			if label=='fake':
				counter+=1
				
	
	# show the output frame and wait for a key press
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	#if 15 frames were detected with spoofed face, send alert
	if counter>40:
		print("Intruder detected. Sending alert to Owner!!")
		alert(frame)
		counter=0

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
