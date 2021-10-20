import cv2
import face_recognition
import pickle#정보를 시리얼라이즈 쭉일렬로 만든뒤 디시리얼라이즈 데이터로 변경
import time
import numpy as np
from tracking import Tracker, Trackable
from collections import Counter

file_name = './image/123.mp4'
frame_count = 0
encoding_file = 'encodings1.pickle'
unknown_name = 'Unknown'
model_method = 'cnn'
min_directions = 7
frame_size = 416

height = 0
width = 0


count_limit = 0
right_count = 0
left_count = 0
direction = ''

trackers = []
trackables = {}
max_name = []
lists = []
# csrt
#tracker = cv2.TrackerCSRT_create()
# kcf
#tracker = cv2.TrackerKCF_create()
# boosting
# tracker = cv2.TrackerBoosting_create()
# mil
# tracker = cv2.TrackerMIL_create()
# tld
# tracker = cv2.TrackerTLD_create()
# medianflow
# tracker = cv2.TrackerMedianFlow_create()
# mosse
# tracker = cv2.TrackerMOSSE_create()

#face_cascade_name = './haarcascades/haarcascade_frontalface_alt.xml'
#face_cascade = cv2.CascadeClassifier()
#if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    #print('### Error loadiqng face cascade ###')
    #exit(0)


data = pickle.loads(open(encoding_file, "rb").read())

tracker = Tracker()

vs = cv2.VideoCapture(file_name)

while True:

        ret, frame = vs.read()
        if frame is None:
            print('### No more frame ###')
            break
        start_time = time.time()
        frame_count += 1
        (height, width) = frame.shape[:2] 
        count_limit = width // 2
        
        # draw a horizontal line in the center of the frame
        cv2.line(frame, (count_limit, 0), (count_limit, height), (0, 255, 255), 2)

        
        frame_mode = 'Detection'
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#rgb로 컨버터
        boxes = face_recognition.face_locations(rgb,
            model=model_method)#가저온 얼굴들을 인코딩
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        
        
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],#데이타 값과 지금 가져온 내용을 비교
                encoding)
            name = unknown_name#안맞을 경우

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:#어떤 이름으로 매치가 됬냐
                    name = data["names"][i] 
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)#데이타가 손이나 테디냐 두개로 인식하면 가장 많이 인식된 걸로
            
            # update the list of names
            names.append(name)
        rects = []

        # loop over the recognized faces 박스엔 4개, 네임엔 네임스
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image
            y = top - 15 if top - 15 > 15 else top + 15
            color = (0, 255, 0)
            line = 2
            if(name == unknown_name):
                color = (0, 0, 255)
                line = 1
                name = ''
            rects.append([left, top, right, bottom])
            cv2.rectangle(frame, (left, top), (right, bottom), color, line)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, color, line)

    

        #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame_gray = cv2.equalizeHist(frame_gray)            
        #faces = face_cascade.detectMultiScale(frame_gray)
        #for (x,y,w,h) in faces:
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 4)

        objects = tracker.update(rects)
    

        for (objectID, centroid) in objects.items():
                # check if a trackable object exists with the object ID
                trackable = trackables.get(objectID, None)

                if trackable is None:
                        trackable = Trackable(objectID, centroid)
                else:
                        lists.append(name)
                        len_list = len(lists)
                        print('lists :',lists)
                        if len_list > min_directions:
                            max_name = max(lists,key=lists.count)
                            lists.pop(0)
                        #평균값을 구한다.
                        x = [c[0] for c in trackable.centroids]
                        variation = centroid[0] - np.mean(x)
                        trackable.centroids.append(centroid)
                        if variation < 0:
                            direction = 1
                        else: 
                            direction = 0
                        trackable.directions.append(direction)
                        mean_directions = int(round(np.mean(trackable.directions)))
                        len_directions = len(trackable.directions)
                        #대세가 내려가는지 올라가는지

                        # check to see if the object has been counted or not
                        if (not trackable.counted) and (len_directions > min_directions):
                                if direction == 1 and centroid[0] < count_limit:
                                        print('left_count')
                                        print(centroid[0],',,',count_limit)
                                        left_count += 1
                                        trackable.counted = True
                                elif direction == 0 and centroid[0] > count_limit:
                                        print('right_count')
                                        print(centroid[0],',,',count_limit)
                                        right_count += 1
                                        trackable.counted = True

                # store the trackable object in our dictionary
                trackables[objectID] = trackable
                text = "ID:{}".format(objectID)
                text_name = "Name:{}".format(max_name)
                cv2.putText(frame, text, (centroid[0] + 10, centroid[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, text_name, (centroid[0] + 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(max_name)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                

        info = [
            ("right",right_count),
            ("left", left_count),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, height - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    

        cv2.imshow("Frame", frame)
        frame_time = time.time() - start_time
        print("[{}] Frame {} time {}".format(frame_mode, frame_count, frame_time))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

vs.release()
cv2.destroyAllWindows()
