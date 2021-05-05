import cv2
import face_recognition
import numpy as np
import pickle
import time

min_confidence = 0.77
width = 1000
height = 0
show_ratio = 1.0
title_name = 'Custom Yolo'


# Load Yolo
net = cv2.dnn.readNet("model/custom-train-yolo_final.weights", "model/custom-train-yolo.cfg")

classes = []
with open("./certificate_dataset/classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
color_lists = np.random.uniform(0, 255, size=(len(classes), 3))

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


file_name = "image/4.jpg"
encoding_file = 'encodings.pickle12'
unknown_name = 'Unknown'
# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'cnn'
output_name = 'video/output_' + model_method + '112' +'.avi'

def detectAndDisplay(image):
    
    start_time = time.time()
    h, w = image.shape[:2]
    height = int(h * width / w)
    image = cv2.resize(image, (width, height))
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    confidences = []
    names = []
    boxes = []
    colors = []

    for out in outs:#식별한걸 하나하나 가져와 돌림
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > min_confidence:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                names.append(classes[class_id])
                colors.append(color_lists[class_id])
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = '{} {:,.2%}'.format(names[i], confidences[i])
            label1 = '{}'.format(names[i])
            color = colors[i]
            print(i, label, x, y, w, h)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), font, 1, color, 2)

    ####################################################################

    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    boxes2 = face_recognition.face_locations(rgb,
        model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes2)

    # initialize the list of names for each face detected
    names1 = []
    

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = unknown_name

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)
        
        # update the list of names
        names1.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes2, names1):
        # draw the predicted face name on the image
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        if(name == unknown_name):
            color = (0, 0, 255)
            line = 1
            name = ''
            
        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            2, color, line)


        if(label1 == 'certificate_kim' and name == 'wang'):
            cv2.putText(image, 'Certified person', (10, height - ((1 * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        else:
            cv2.putText(image, 'i dont know', (10, height - ((2 * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    

                
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    # show the output image
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    cv2.imshow("Recognition", image)

    
    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    global writer
    if writer is None and output_name is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_name, fourcc, 24,
                (image.shape[1], image.shape[0]), True)

    # if the writer is not None, write the 5frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(image)
##########################################################3        
# load the known faces and embeddings
data = pickle.loads(open(encoding_file, "rb").read())

#-- 2. Read the video stream
cap = cv2.VideoCapture(file_name)
writer = None
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        # close the video file pointers
        cap.release()
        # close the writer point
        writer.release()
        break
    detectAndDisplay(frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()




