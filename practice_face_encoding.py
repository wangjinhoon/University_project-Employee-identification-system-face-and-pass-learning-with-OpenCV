import cv2
import face_recognition
import pickle

dataset_paths = ['./dataset/jang-align/','./dataset/kim-align/','./dataset/song-align/','./dataset/tedy-align/','./dataset/unknown-align/','./dataset/wang-align/']
names = ['jang','kim','song','tedy','unknown','wang']
number_images = 10#이미지 갯수
image_type = '.jpg'
encoding_file = 'encodings.pickle1'#pickle을 이용해 만듬 
# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'cnn'#대표적인 시각화 방법 대신 성능이 느림,HOG 아날로그 그림을 디지털화 하는것 빠른 대신 정확도 떨어짐


# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []#두가지 배열을 만듬

# loop over the image paths
for (i, dataset_path) in enumerate(dataset_paths):#데이타 패스를 하나씩 돌림
    # extract the person name from names
    name = names[i]

    for idx in range(number_images):#사진을 하나씩 돌리며 벡터 128베쉬d값을 구함
        file_name = dataset_path + str(idx+1) + image_type

        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(file_name)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,#여기서 얼굴을 찾는데 cnn 방식으로 함
            model=model_method)

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)#인코딩 작업

        # loop over the encodings
        for encoding in encodings:#128개의 리얼넘버를 하나씩 인코딩 해줌
            # add each encoding + name to our set of known names and
            # encodings
            print(file_name, name, encoding)
            knownEncodings.append(encoding)
            knownNames.append(name)
        
# Save the facial encodings + names to disk#pickle file에 저장
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encoding_file, "wb")
f.write(pickle.dumps(data))
f.close()
print("end")


