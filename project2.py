import cv2
import face_recognition
import numpy as np
import pickle
import time
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import tkinter.scrolledtext as tkst


min_confidence = 0.85
width = 1000
height = 0
show_ratio = 1.0
title_name = 'project Yolo'

file_name = "image/6.avi"
file_name1 = "image/127.jpg"
weight_name = "model/custom-train-yolo_3000.1.weights"
cfg_name = "model/custom-train-yolo.cfg"
classes_name = "./certificate_dataset/classes.names"
encoding_file = 'encodings.pickle1'
title_name = "project Yolo"
cap = cv2.VideoCapture()
writer = None
unknown_name = 'Unknown'
# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'cnn'
# load the known faces and embeddings
data = pickle.loads(open(encoding_file, "rb").read())

#output_name = 'video/output_' + model_method + '7' +'.avi'
#writer = None

cap = cv2.VideoCapture()
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
    file_name =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("jpeg files","*.jpg"),("avi","*.avi"),("all files","*.*")))
    print('File name : ', file_name)
    global cap
    cap = cv2.VideoCapture(file_name)
    detectAndDisplay()


def detectAndDisplay():
    start_time = time.time()
    _, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        # close the video file pointers
        cap.release()
        # close the writer point
        writer.release()

    (h, w) = frame.shape[:2]
    height = int(h * width / w)
    frame = cv2.resize(frame, (width, height))
    # Load Yolo
    net = cv2.dnn.readNet(weight_name, cfg_name)
    classes = []
    with open(classes_name, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    color_lists = np.random.uniform(0, 255, size=(len(classes), 3))

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    label1=''

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)

    ####################################################################

    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    boxes2 = face_recognition.face_locations(rgb,
        model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes2)

    # initialize the list of names for each face detected
    names1 = []
    names2 = []
    

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
        print(names1)

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
        names2.append(name)
            
        cv2.rectangle(frame, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            2, color, line)
        


    if(label1 == 'certificate_wang' and names2.count('wang')>1):
        cv2.putText(frame, 'Certified_person', (10, height - ((1 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, '           ', (10, height - ((2 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    elif(label1 == 'certificate_song' and names2.count('song')>1):
        cv2.putText(frame, '                ', (10, height - ((1 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, 'Certified_person', (10, height - ((2 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    elif(label1 == 'certificate_kim' and names2.count('kim')>1):
        cv2.putText(frame, '                ', (10, height - ((1 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, 'Certified_person', (10, height - ((2 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    elif(label1 == 'certificate_jang' and names2.count('jang')>1):
        cv2.putText(frame, '                ', (10, height - ((1 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, 'Certified_person', (10, height - ((2 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    elif(label1 == '' and names2.count('wang')>1):
        cv2.putText(frame, '                ', (10, height - ((1 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, 'show id_card', (10, height - ((2 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    else:
        cv2.putText(frame, '                ', (10, height - ((1 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, 'i dont know', (10, height - ((2 * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    
    #if the video writer is None *AND* we are supposed to write
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
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(100, detectAndDisplay)
    print('--------')

##########################################################3


main = Tk()
main.title(title_name)
main.geometry()


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
imageFrame = Frame(main) # 프레임 너비, 높이 설정
imageFrame.grid(row=5,column=0,columnspan=4) # 격자 행, 열 배치

lmain=Label(imageFrame)
lmain.grid(row=0, column=0)

main.mainloop()



