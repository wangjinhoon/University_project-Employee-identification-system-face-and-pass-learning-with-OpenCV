import cv2
import face_recognition
import pickle
import time

file_name = '3.JPG'
encoding_file = 'encodings.pickle1'
unknown_name = 'Unknown'
# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'cnn'
output_name = 'video/output_' + model_method + '2' +'.avi'

def detectAndDisplay(image):
    start_time = time.time()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    boxes = face_recognition.face_locations(rgb,
        model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    # initialize the list of names for each face detected
    names = []

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
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
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



