import numpy as np
import dlib
import cv2

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

dataset_paths = [ './dataset/jang-fornt/','./dataset/kim-front/','./dataset/song-front/','./dataset/tedy-front/','./dataset/unknown-front/','./dataset/wang-front/']
output_paths = [ './dataset/jang-align/','./dataset/kim-align/','./dataset/song-align/','./dataset/tedy-align/','./dataset/unknown-align/', './dataset/wang-align/']
number_images = 10
image_type = '.jpg'

predictor_file = './model/shape_predictor_68_face_landmarks.dat'
MARGIN_RATIO = 1.5
OUTPUT_SIZE = (300, 300)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

def getFaceDimension(rect):
    return (rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top())

def getCropDimension(rect, center):
    width = (rect.right() - rect.left())
    half_width = width // 2
    (centerX, centerY) = center
    startX = centerX - half_width
    endX = centerX + half_width
    startY = rect.top()
    endY = rect.bottom() 
    return (startX, endX, startY, endY)    

for (i, dataset_path) in enumerate(dataset_paths):
    output_path = output_paths[i]
    
    for idx in range(number_images):
        input_file = dataset_path + str(idx+1) + image_type

        # get RGB image from BGR, OpenCV format
        image = cv2.imread(input_file)
        image_origin = image.copy()

        (image_height, image_width) = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 1)

        for (i, rect) in enumerate(rects):
            (x, y, w, h) = getFaceDimension(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            points = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])
            show_parts = points[EYES]

            right_eye_center = np.mean(points[RIGHT_EYE], axis = 0).astype("int")
            left_eye_center = np.mean(points[LEFT_EYE], axis = 0).astype("int")

            eye_delta_x = right_eye_center[0,0] - left_eye_center[0,0]
            eye_delta_y = right_eye_center[0,1] - left_eye_center[0,1]
            degree = np.degrees(np.arctan2(eye_delta_y,eye_delta_x)) - 180

            eye_distance = np.sqrt((eye_delta_x ** 2) + (eye_delta_y ** 2))
            aligned_eye_distance = left_eye_center[0,0] - right_eye_center[0,0]
            scale = aligned_eye_distance / eye_distance

            eyes_center = ((left_eye_center[0,0] + right_eye_center[0,0]) // 2,
                    (left_eye_center[0,1] + right_eye_center[0,1]) // 2)
                    
            metrix = cv2.getRotationMatrix2D(eyes_center, degree, scale)

            warped = cv2.warpAffine(image_origin, metrix, (image_width, image_height),
                flags=cv2.INTER_CUBIC)

            (startX, endX, startY, endY) = getCropDimension(rect, eyes_center)

            croped = warped[startY:endY, startX:endX]
            output = cv2.resize(croped, OUTPUT_SIZE)
            #output = warped[startY:endY, startX:endX]
            
            output_file = output_path + str(idx+1) + image_type
            cv2.imshow(output_file, output)
            cv2.imwrite(output_file, output)
        
cv2.waitKey(0)   
cv2.destroyAllWindows()
