import numpy as np
import cv2
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import tkinter.scrolledtext as tkst

min_confidence = 0.5
width = 800
height = 0
show_ratio = 1.0
file_name = "./custom/fruit10.jpg"
weight_name = "./model/custom-train-yolo_final.weights"
cfg_name = "./custom/custom-train-yolo.cfg"
classes_name = "./custom/classes.names"
title_name = 'Custom Yolo'
classes = []

def selectWeightFile():
    global weight_name
    weight_name =  filedialog.askopenfilename(initialdir = "./model",title = "Select Weight file",filetypes = (("weights files","*.weights"),("all files","*.*")))
    weight_path['text'] = weight_name 

def selectCfgFile():
    global cfg_name
    cfg_name =  filedialog.askopenfilename(initialdir = "./",title = "Select Cfg file",filetypes = (("cfg files","*.cfg"),("all files","*.*")))
    cfg_path['text'] = cfg_name

def selectClassesFile():
    global classes_name
    classes_name =  filedialog.askopenfilename(initialdir = "./",title = "Select Classes file",filetypes = (("names files","*.names"),("all files","*.*")))
    classes_path['text'] = classes_name

def selectFile():
    file_name =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    read_image = cv2.imread(file_name)
    file_path['text'] = file_name
    detectAndDisplay(read_image)

def detectAndDisplay(image):
    net = cv2.dnn.readNet(weight_name, cfg_name)
    with open(classes_name, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    color_lists = np.random.uniform(0, 255, size=(len(classes), 3))
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    h, w = image.shape[:2]
    height = int(h * width / w)
    img = cv2.resize(image, (width, height))

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
            color = colors[i]
            print(i, label, x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
            
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detection_image.config(image=imgtk)
    detection_image.image = imgtk
    
main = Tk()
main.title(title_name)
main.geometry()

read_image = cv2.imread(file_name)
image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=image)

label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

weight_title = Label(main, text='Weight')
weight_title.grid(row=1,column=0,columnspan=1)
weight_path = Label(main, text=weight_name)
weight_path.grid(row=1,column=1,columnspan=2)
Button(main,text="Select", height=1,command=lambda:selectWeightFile()).grid(row=1, column=3, columnspan=1, sticky=(N, S, W, E))

cfg_title = Label(main, text='Cfg')
cfg_title.grid(row=2,column=0,columnspan=1)
cfg_path = Label(main, text=cfg_name)
cfg_path.grid(row=2,column=1,columnspan=2)
Button(main,text="Select", height=1,command=lambda:selectCfgFile()).grid(row=2, column=3, columnspan=1, sticky=(N, S, W, E))

classes_title = Label(main, text='Classes')
classes_title.grid(row=3,column=0,columnspan=1)
classes_path = Label(main, text=classes_name)
classes_path.grid(row=3,column=1,columnspan=2)
Button(main,text="Select", height=1,command=lambda:selectClassesFile()).grid(row=3, column=3, columnspan=1, sticky=(N, S, W, E))

file_title = Label(main, text='Image')
file_title.grid(row=4,column=0,columnspan=1)
file_path = Label(main, text=file_name)
file_path.grid(row=4,column=1,columnspan=2)
Button(main,text="Select", height=1,command=lambda:selectFile()).grid(row=4, column=3, columnspan=1, sticky=(N, S, W, E))

detection_image=Label(main, image=imgtk)
detection_image.grid(row=5,column=0,columnspan=4)

detectAndDisplay(read_image)

main.mainloop()
