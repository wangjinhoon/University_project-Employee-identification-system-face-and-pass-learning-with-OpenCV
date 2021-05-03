import cv2
import numpy as np
import time

min_confidence = 0.5
width = 800
height = 0
show_ratio = 1.0
file_name = "./custom/fruit10.jpg"

# Load Yolo
net = cv2.dnn.readNet("./model/custom-train-yolo_final.weights", "./custom/custom-train-yolo.cfg")
classes = []
with open("./custom/classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)
color_lists = np.random.uniform(0, 255, size=(len(classes), 3))

layer_names = net.getLayerNames()
# print(layer_names)
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# print(net.getUnconnectedOutLayers())
print(output_layers)

start_time = time.time()
img = cv2.imread(file_name)
h, w = img.shape[:2]
height = int(h * width / w)
print(height, width)


img = cv2.resize(img, (width, height))
cv2.imshow("Original - "+file_name, img)

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

confidences = []
names = []
boxes = []
colors = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > min_confidence:
            print(detection)
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
            names.append(classes[class_id])
            colors.append(color_lists[class_id])

indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = '{} {:,.2%}'.format(names[i], confidences[i])
        color = colors[i]
        print(i, label, color, x, y, w, h)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

cv2.imshow("Custom Yolo - "+file_name, img)

end_time = time.time()
process_time = end_time - start_time
print("=== A frame took {:.3f} seconds".format(process_time))

cv2.waitKey(0)
cv2.destroyAllWindows()
