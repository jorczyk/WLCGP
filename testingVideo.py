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

        result = wlcgpt.testWlcgp(stream_reader.current_frame, numx, numy, base, E, P, nTrain, NumPerClassTrain)

        # TU WSTAWIC TO Z WLCGP
        # img2 = stream_reader.current_frame
        # img2 = ((img2 - np.mean(img2)) + 128) / np.std(img2) * 20
        # iiCell = commons.block(numx, numy, img2)
        # lbpII = []
        # lbpII = np.asarray(lbpII)
        #
        # for k in range(numx):
        #     for m in range(numy):
        #         iiCellBlock = iiCell[k, m]  # k,m
        #         blockLBPII = wlcgpFile.wlcgp(iiCellBlock)
        #         blockLBPII = np.transpose(blockLBPII)
        #         if (m == 0) & (k == 0):
        #             lbpII = blockLBPII
        #         else:
        #             lbpII = np.concatenate((lbpII, blockLBPII))  # LBP_I=[LBP_I,Block_LBP_I]; moze byc concatenate
        #             # end
        # # end
        # lbpII = lbpII.reshape(1, lbpII.shape[0])
        # tcoor = lbpII.dot(base)  # lbpII * base// base trzeba tu liczyc!!! //rozmiar ok
        #
        # # print np.transpose(E).shape #ok
        # # print np.transpose(tcoor).shape #ok
        #
        # tcoor = np.transpose(E).dot(np.transpose(tcoor))  # E trzeba bedzie tu liczyc? ale z czego?
        #
        # # print tcoor.shape #ok
        # # k = 1
        # # mdist = []
        # # while k <= nTrain:  # co z tym nTrain???
        # #     k += 1
        # mdist = [None] * nTrain
        # for k in range(nTrain):
        #     mdist[k] = np.linalg.norm(tcoor - P[:, k])  # P trzeba wyciagac
        # # end
        #
        # ####################################
        #
        # # 3 NN algorithm
        # # dist, index2 = np.sort(mdist)
        # index2 = np.argsort(mdist)
        # dist = mdist.sort
        #
        # class1 = int(
        #     np.math.floor(index2[1] / NumPerClassTrain - 0.1) + 1)  # NumPerClassTrain - skad to ma byc potem brane?
        # class2 = int(np.math.floor(index2[2] / NumPerClassTrain - 0.1) + 1)
        # class3 = int(np.math.floor(index2[3] / NumPerClassTrain - 0.1) + 1)
        #
        # # print index2
        #
        # if (class1 != class2) & (class2 != class3):
        #     result = class1
        # else:
        #     if class1 == class2:
        #         result = class1
        #     if class2 == class3:
        #         result = class2

                # KONIEC WLCGP
                #########################################


                #     [label, confidence] = face_recognizer.predict(np.asarray(face_image))
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
