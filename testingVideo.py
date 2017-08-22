import time
import numpy as np
import cv2
import commons
import facecropper
import streamreader
import wlcgpTesting as wlcgpt

##################################


stream_reader = streamreader.Stream("1")

try:
    face_cropper = facecropper.FaceCropper()
except:
    raise

basePath = './train/base.npy'
variablesPath = './train/variables.npy'
EPath = './train/E.npy'
PPath = './train/P.npy'

base = np.load(basePath)
trainVariables = np.load(variablesPath)
E = np.load(EPath)
P = np.load(PPath)

# NumPerson = trainVariables[0]
# NumPerClass = trainVariables[1]
NumPerClassTrain = trainVariables[2]
numx = trainVariables[3]
numy = trainVariables[4]
nTrain = trainVariables[5]
##################################


while (True):
    begin_time = time.time()
    try:
        gray_frame = stream_reader.read()
    except IOError:
        print "Something went wrong with stream"
        break
    except:
        print "End of stream"
        break
    face_images = face_cropper.get_face_images(gray_frame)
    face_locations = face_cropper.get_face_locations()
    for face_image, (x, y, w, h) in zip(face_images, face_locations):
        # print face_image.shape
        result = wlcgpt.testWlcgp(face_image, numx, numy, base, E, P, nTrain, NumPerClassTrain)
        print result

        # KONIEC WLCGP
        #########################################

        cv2.rectangle(
            stream_reader.current_frame,
            (x, y), (x + w, y + h), (255, 255, 255), 1)
        #     if label != -1:
        #         cv2.putText(
        #             stream_reader.current_frame,
        #             "ID: " + str(label) + " , conf.: " + str(round(confidence)),
        #             (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
        # if args.reference_faces_path:
        #     cv2.imshow("mlnSpyHole - face window", reference_faces[label])
        # print "Detected id: " + str(label), \
        #     "\b, conf.: " + str(round(confidence)), \
        #     "\b, at: " + str(x) + ' ' + str(y), \
        #     "- " + str(x + w) + ' ' + str(y + h)

    finish_time = time.time()

    cv2.putText(
        stream_reader.current_frame,
        "FPS: " + str(round(1 / (finish_time - begin_time), 1)),
        (0, 12),
        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    cv2.imshow("mlnSpyHole - main window", stream_reader.current_frame)

    # print "Result: " + str(result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
