# importer les packages et librairies necessaires
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import paho.mqtt.client as mqtt

def on_publish(client,userdata,result):
        print("data published")
        pass

client1=mqtt.Client()
client1.on_publish = on_publish
client1.connect("192.168.43.205",1883,60)


def sound_alarm(path):
	# Executer le fichier audio
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# calculer les distances euclidiennes entre les deux
	# ensembles de points de repère verticaux 
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# calculer les distances euclidiennes entre les deux
	# ensembles de points de repère horizontaux
	C = dist.euclidean(eye[0], eye[3])

	# calculer l'eye aspect ratio
	ear = (A + B) / (2.0 * C)

	return ear
 
# Le parsing
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="chemin vers le prédicteur de repère facial")
ap.add_argument("-a", "--alarm", type=str, default="",
	help="Chemin vers le fichier audio")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="Webcam index")
args = vars(ap.parse_args())
 
# définir deux constantes, une pour l'EAR pour indiquer la distraction des yeux
# puis une deuxième constante pour le nombre d'images consécutives
# l'œil doit être inférieur au seuil pour déclencher l'alarme
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 5

# initialiser le compteur des frames et un boolean pour indiquer si l'alarme est declenche ou non
COUNTER = 0
ALARM_ON = False

# Detecter le visage avec dlib 
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Prend les indexes des yeux 
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

# boucle sur les images du flux vidéo
while True:
	# Prend l'image et changer sa taille et la convertir 
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detecter les visages
	rects = detector(gray, 0)

	# Boucle sur les visages
	for rect in rects:
		# Detection des reperes faciaux et convertir en un tableau
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# Extraire les yeux et calculer son EAR
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# calculer EAR final
		ear = (leftEAR + rightEAR) / 2.0

		# Visualiser les yeux
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# Si le ear est moins que la limite on demarre le compteur
		if ear < EYE_AR_THRESH:
			COUNTER += 1

			# si le compteur depasse un temps
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# On declenche l'alarme
				if not ALARM_ON:
					ALARM_ON = True

					# Par la lecture du fichier audio
					if args["alarm"] != "":
						t = Thread(target=sound_alarm,
							args=(args["alarm"],))
						t.deamon = True
						t.start()

				# Ecrire dans l'image
				cv2.putText(frame, "Attention!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				client1.publish("esp8266/c","1")
		# Sinon on initialise le compteur a 0
		else:
			COUNTER = 0
			ALARM_ON = False
			client1.publish("esp8266/c","0")

		# Afficher le ear calculé
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	
	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()
