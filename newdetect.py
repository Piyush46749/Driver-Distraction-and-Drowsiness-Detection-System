from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import playsound

# calculating eye aspect ratio
def eye_aspect_ratio(eye):

    # compute the vertical euclidean distances

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the horizontal euclidean distances

    C = dist.euclidean(eye[0], eye[3])

    #eye aspect ratio

    ear = (A + B) / (2.0 * C)
    return ear


#mouth aspect ratio

def mouth_aspect_ratio(mou):

    # horizontal euclidean distances

    X = dist.euclidean(mou[0], mou[6])

    # vertical euclidean distances

    Y1 = dist.euclidean(mou[2], mou[10])
    Y2 = dist.euclidean(mou[4], mou[8])

    # taking average

    Y = (Y1 + Y2) / 2.0

    #mouth aspect ratio

    mar = Y / X
    return mar

JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

# define constants for aspect ratios

EYE_AR_THRESH = 0.25 #threshold of eye aspect ratio)
EYE_AR_CONSEC_FRAMES = 48 #48 frames per second
MOU_AR_THRESH = 0.40

COUNTER = 0 #blink counter
yawnStatus = False #starting status false
yawns = 0 #yawn counter

def alert_driver():
    print ("\tDrowsiness detected, waking the driver up..")
    playsound.playsound("alarm.wav")

 
# Create the haar cascade
cascPath = 'haarcascade_frontalface_default.xml'  #pre trained haarcascades
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

faceCascade = cv2.CascadeClassifier(cascPath)

predictor = dlib.shape_predictor(PREDICTOR_PATH)


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
# print(lStart,lEnd)
# print(rStart,rEnd)
# print(mStart,mEnd)


cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_yawn_status = yawnStatus

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05,
            minNeighbors=5, minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE)

    print('Found {0} faces!'.format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0xff, 0), 2)


        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
        landmarks_display = landmarks[RIGHT_EYE_POINTS
                + LEFT_EYE_POINTS + MOUTH_OUTLINE_POINTS
                + MOUTH_INNER_POINTS + JAWLINE_POINTS]

         #   print(RIGHT_EYE_POINTS)

        for (idx, point) in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 2, color=(0, 0xff, 0xff), thickness=-1)

      
        shape = predictor(gray, dlib_rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouEAR = mouth_aspect_ratio(mouth)

        # print(mouEAR)
        # average the eye aspect ratio together for both eyes

        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0xff, 0xff), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0xff, 0xff), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 0xff, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            cv2.putText(
                frame,
                'Eyes Closed ',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0xff),
                2,
                )

                # if the eyes were closed for a sufficient number of

            if COUNTER >= EYE_AR_CONSEC_FRAMES:

                    # draw an alarm on the frame

                cv2.putText(
                    frame,
                    alert_driver(),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0xff),
                    2,
                    )
        else:

            # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm

            COUNTER = 0
            cv2.putText(
                frame,
                'Eyes Open ',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0xff, 0),
                2,
                )
        cv2.putText(
            frame,
            'EAR: {:.2f}'.format(ear),
            (480, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0xff),
            2,
            )

        # yawning detections

        if mouEAR > MOU_AR_THRESH:
            cv2.putText(
                frame,
                'Yawning ',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0xff),
                2,
                )
            yawnStatus = True
            output_text = 'Yawn Count: ' + str(yawns + 1)
            cv2.putText(
                frame,
                output_text,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0xff, 0, 0),
                2,
                )
        else:
            yawnStatus = False
        if prev_yawn_status == True and yawnStatus == False:
            yawns += 1

        cv2.putText(
            frame,
            'MAR: {:.2f}'.format(mouEAR),
            (480, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0xff),
            2,
            )
        cv2.putText(
            frame,
            'Drowsiness Detection',
            (370, 470),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (27, 200, 102),
            1,
            )

    cv2.imshow('Landmarks found',frame)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()