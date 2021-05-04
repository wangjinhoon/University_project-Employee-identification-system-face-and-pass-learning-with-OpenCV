import cv2
import time

file_name = './video/face_01.mp4'
frame_count = 0


# csrt
tracker = cv2.TrackerCSRT_create()
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

face_cascade_name = './haarcascades/haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('### Error loadiqng face cascade ###')
    exit(0)

detected = False
frame_mode = 'Tracking'
elapsed_time = 0

trackers = cv2.MultiTracker_create()#MultiTracker

vs = cv2.VideoCapture(0)

while True:
        ret, frame = vs.read()
        if frame is None:
            print('### No more frame ###')
            break
        start_time = time.time()
        frame_count += 1
        if detected:
            frame_mode = 'Tracking'
            (success, boxes) = trackers.update(frame)
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            frame_mode = 'Detection'
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)            
            faces = face_cascade.detectMultiScale(frame_gray)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 4)

            trackers.add(tracker, frame, tuple(faces[0]))
            detected = True

        cv2.imshow("Frame", frame)
        frame_time = time.time() - start_time
        elapsed_time += frame_time
        print("[{}] Frame {} time {}".format(frame_mode, frame_count, frame_time))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

print("Elapsed time {}".format(elapsed_time))
vs.release()
cv2.destroyAllWindows()
