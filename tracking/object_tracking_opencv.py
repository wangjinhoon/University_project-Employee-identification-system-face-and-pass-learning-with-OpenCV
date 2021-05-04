import cv2
print(cv2.__version__)
import time

file_name = './video/passenger_01.mp4'
frame_count = 0

trackers = cv2.MultiTracker_create()

vs = cv2.VideoCapture(file_name)

while True:
        ret, frame = vs.read()
        if frame is None:
                print('### No more frame ###')
                break
        start_time = time.time()
        frame_count += 1

        (success, boxes) = trackers.update(frame)

        for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        frame_time = time.time() - start_time 
        print("Frame {} time {}".format(frame_count, frame_time))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
                box = cv2.selectROI("Frame", frame, fromCenter=False,
                        showCrosshair=True)
                # csrt
                #tracker = cv2.TrackerCSRT_create()
                # kcf
                tracker = cv2.TrackerKCF_create()
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

                trackers.add(tracker, frame, box)

        elif key == ord("q"):
                break


vs.release()
cv2.destroyAllWindows()
