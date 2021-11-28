import cv2
import face_recognition
import numpy as np
import pickle
import time
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
from tracking import Tracker, Trackable
import tkinter.scrolledtext as tkst


min_confidence = 0.85
width = 800
height = 0
show_ratio = 1.0
title_name = 'Wangs_project'
frame_count = 0
file_name = "image/.avi"
weight_name = "model/custom-train-yolo_4.1.weights"
cfg_name = "model/custom-train-yolo.cfg"
classes_name = "./certificate_dataset/classes.names"
encoding_file = 'encodings.pickle1'
cap = cv2.VideoCapture()
writer = None
unknown_name = 'Unknown'
# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'cnn'
min_directions = 5
man_directions = 10
# load the known faces and embeddings
data = pickle.loads(open(encoding_file, "rb").read())
a = 'Certified_person'
b = 0
#b1 = ''
c = 0
#t = ''
#r = ''
#l = ''
wang = 0
song = 0
jang = 0
kim = 0
maximum_name = []

height = 0
width = 0

count_limit = 0
right_count = 0
left_count = 0
direction = ''
z = 0
tracker = Tracker()
trackers = []
trackables = {}
max_name = []
max_name1 = []
lists = []



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

def selectEncoding_file():
    global Encoding_file
    Encoding_file =  filedialog.askopenfilename(initialdir = "./",title = "Select Encoding file",filetypes = (("names files","*.pickle"),("all files","*.*")))
    Encoding_path['text'] = Encoding_file

def cam():
    global writer
    global cap
    global b
    b=b+1
    writer = None
    file_name = 0
    cap = cv2.VideoCapture(file_name)
    detectAndDisplay()

def selectFile():
    global writer
    global cap
    global b
    file_name =  filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("jpeg files","*.jpg"),("avi","*.avi"),("all files","*.*")))
    file_path['text'] = file_name
    b=b+1
    writer = None
    cap = cv2.VideoCapture(file_name)
    detectAndDisplay()

def detectAndDisplay():
    global a
    global c
    global wang
    global song
    global kim
    global jang
    global frame_count
    global trackers
    global trackables
    global min_directions
    global left_count
    global right_count
    global max_name
    global max_name1
    global z
    global man_directions
    t = None 
    r = None
    global b
    b1 = None
    l = None
    global maximum_name
    name = []
    x = None
    y = None
    left = None
    right = None
    top = None
    bottom = None
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

    global writer
    _, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        # close the video file pointers
        cap.release()
        # close the writer point
        writer.release()
        return
    start_time = time.time()
    frame_count += 1
    (height, width) = frame.shape[:2] 
    count_limit = (width // 2)-200
    count_limit1 = (width // 2)+200

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)


    # draw a horizontal line in the center of the frame
    cv2.line(frame, (count_limit, 0), (count_limit, height), (0, 255, 255), 2)
    cv2.line(frame, (count_limit1, 0), (count_limit1, height), (0, 255, 255), 2)


    
    frame_mode = 'Detection'

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
            print(i, label, x, y, x + w, y + h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)
            

    ####################################################################
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#rgb로 컨버터
    boxes = face_recognition.face_locations(rgb,
        model=model_method)#가저온 얼굴들을 인코딩
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    names1 = []
    c_names = []
    
    
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
    maximum = 0
    # loop over the recognized faces 박스엔 4개, 네임엔 네임스
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        hh = bottom-top
        ww = right-left
        now = hh*ww

        if(name == unknown_name):
            color = (0, 0, 255)
            line = 1
            name = ''
 
        if maximum < now:
            maximum = now
            maximum_name = name
            t,r,b1,l = top, right, bottom, left

        if x != None and y != None and left != None and right != None and top != None and bottom != None:
            if(x < (left+right)/2 < x+w and y < (top+bottom)/2 < y+h):
                c_names.append(name)
                print("c_names",c_names)
        
        
        cv2.rectangle(frame, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.75, color, line)
    
    if l != None and t != None and r != None and b1 != None:
        rects.append([l, t, r, b1])
        del l, t, r, b1
    
    

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
                    
                    if(label1 == 'certificate_wang' and 'wang' in max_name and c_names.count('wang')>0):
                        lists.append('c_wang')
                        len_list = len(lists)
                        if len_list > min_directions:
                            max_name = max(lists,key=lists.count)
                            lists.pop(0)
                            if  'c_wang' in max_name:
                                max_name1 = 'c_wang'
                                c = 1 
                                print('maxname1: ')
                                print(max_name1)
                        

                    
                    elif(label1 == 'certificate_song' and 'song' in max_name and c_names.count('song')>0):
                        lists.append('c_song')
                        len_list = len(lists)
                        if len_list > min_directions:
                            max_name = max(lists,key=lists.count)
                            lists.pop(0)
                            if  'c_song' in max_name:
                                max_name1 = 'c_song'
                                c = 1 
                                print('maxname1: ')
                                print(max_name1)

                    
                    else:
                        if c != 0:
                            max_name = max_name1
                                                   
                        else:
                            lists.append(maximum_name)
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
                    len_directions = len(trackable.directions)
                    #대세가 내려가는지 올라가는지

                    # check to see if the object has been counted or not
                    if (not trackable.counted) and (len_directions > min_directions):
                        
                            if z == 1:
                                        if direction == 0 and centroid[0] > count_limit1:
                                                print('z1111')
                                            
                            else:
                                
                                if direction == 1 and centroid[0] < count_limit:
                                    
                                        if 'c_wang' in max_name:
                                            print('left_count')
                                            print(centroid[0],',,',count_limit)
                                            left_count += 1
                                            trackable.counted = True
                                            log_ScrolledText.insert(END,"왕진훈\n")
                                            log_ScrolledText.insert(END,time.ctime())
                                            log_ScrolledText.insert(END,"\n")
                                            c = 0
                                            max_name = []
                                        
                                        elif 'c_song' in max_name:
                                            print('left_count')
                                            print(centroid[0],',,',count_limit)
                                            left_count += 1
                                            trackable.counted = True
                                            log_ScrolledText.insert(END,"송민수\n")
                                            log_ScrolledText.insert(END,time.ctime())
                                            log_ScrolledText.insert(END,"\n")
                                            c = 0
                                            max_name = []
                                    
                                        else :
                                            z = 1

                                   
                                elif direction == 0 and centroid[0] > count_limit1:
                                        print('right_count')
                                        print(centroid[0],',,',count_limit1)
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

    # if the writer is not None, write the 5frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(frame)
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 

    if z == 1:
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        z = tracker.z

    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(300, detectAndDisplay)


##########################################################3


main = Tk()
main.title(title_name)
main.geometry()


label=Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

label1=Label(main, text="Entrance")
label1.config(font=("Courier", 15))
label1.grid(row=0,column=5,columnspan=1,sticky=(N, S, W, E))

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

file_title1 = Label(main, text='Encoding_file')
file_title1.grid(row=4,column=0,columnspan=1)
Encoding_path = Label(main, text=encoding_file)
Encoding_path.grid(row=4,column=1,columnspan=2)
Button(main,text="Select", height=1,command=lambda:selectEncoding_file()).grid(row=4, column=3, columnspan=1, sticky=(N, S, W, E))

file_title = Label(main, text='Image')
file_title.grid(row=5,column=0,columnspan=1)
file_path = Label(main, text=file_name)
file_path.grid(row=5,column=1,columnspan=2)
Button(main,text="Select", height=1,command=lambda:selectFile()).grid(row=5, column=3, columnspan=1, sticky=(N, S, W, E))

file_title2 = Label(main, text=' Play CAM')
file_title2.grid(row=6,column=2,columnspan=1)
Button(main,text="Select", height=1,command=lambda:cam()).grid(row=6, column=3, columnspan=1, sticky=(N, S, W, E))

sizeLabel=Label(main, text='Min Confidence : ')
sizeLabel.grid(row=6,column=0)
sizeVal  = IntVar(value=min_confidence)
sizeSpin = Spinbox(main, textvariable=sizeVal,from_=0, to=1, increment=0.05, justify=RIGHT)
sizeSpin.grid(row=6, column=1)

log_ScrolledText = tkst.ScrolledText(width=10, height=10)
log_ScrolledText.grid(row=1,column=5,rowspan=5)

log_ScrolledText.configure(font='TkFixedFont')

imageFrame = Frame(main) # 프레임 너비, 높이 설정
imageFrame.grid(row=7,column=0,columnspan=4) # 격자 행, 열 배치

lmain=Label(imageFrame)
lmain.grid(row=0, column=0)

main.mainloop()



