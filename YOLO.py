import cv2
import numpy as np

min_confidence = 0.7

# Load Yolo
net = cv2.dnn.readNet("model/custom-train-yolo_final.weights", "model/custom-train-yolo.cfg")

with open("certificate_dataset/classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("./image/17.JPG")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape
cv2.imshow("Original Image", img)

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                                            #픽셀 단위
net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > min_confidence:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)#그림을 그리다 보면 노이즈가 생김 그걸 없애주는 함수
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:#박스 수만큼 돌려서
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(i, label)
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, (0, 255, 0), 1)

cv2.imshow("YOLO Image", img)
cv2.waitKey('q')
cv2.destroyAllWindows()

            
