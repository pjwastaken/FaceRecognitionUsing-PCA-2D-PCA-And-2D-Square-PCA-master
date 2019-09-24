import cv2
from algoClass import my_algo
from imageMatrix import imageToMatrixClass
from dataset import datasetClass

#reco_type = "image"
reco_type = "group"
#reco_type = "video"

no_of_images_of_one_person = 8
dataset_obj = datasetClass(no_of_images_of_one_person)

images_paths_for_training = dataset_obj.images_path_for_training
labels_for_training = dataset_obj.labels_for_training
no_of_elements_for_training = dataset_obj.no_of_images_for_training

images_paths_for_testing = dataset_obj.images_path_for_testing
labels_for_testing = dataset_obj.labels_for_testing
no_of_elements_for_testing = dataset_obj.no_of_images_for_testing

images_targests = dataset_obj.images_target

img_width, img_height = 50, 50
imageToMatrixClassObj = imageToMatrixClass(images_paths_for_training, img_width, img_height)
img_matrix = imageToMatrixClassObj.get_matrix()

my_algo_class_obj = my_algo(img_matrix, labels_for_training, images_targests, no_of_elements_for_training, img_width, img_height, quality_percent=90)
new_coordinates = my_algo_class_obj.reduce_dim()

#org_img = my_algo_class_obj.new_to_old_cords(new_coordinates[:, 2])
#my_algo_class_obj.show_image("Original Image", org_img)

#my_algo_class_obj.show_eigen_faces(50, 200, 0)
#my_algo_class_obj.show_eigen_faces(50, 200, 1)
#my_algo_class_obj.show_eigen_faces(50, 200, 2)
#my_algo_class_obj.show_eigen_faces(50, 200, 3)


if reco_type is "image":
    correct = 0
    wrong = 0
    i = 0

    for img_path in images_paths_for_testing:
        img = my_algo_class_obj.img_from_path(img_path)
        my_algo_class_obj.show_image("Recognize Image", img)
        new_cords_for_image = my_algo_class_obj.new_cords(img)

        finded_name = my_algo_class_obj.recognize_face(new_cords_for_image)
        target_index = labels_for_testing[i]
        original_name = images_targests[target_index]

        if finded_name is original_name:
            correct += 1
            print("Correct Result", "Name:", finded_name)
        else:
            wrong += 1
            print("Wrong Result", "Name:", finded_name)
        i += 1

    print("Total Correct", correct)
    print("Total Wrong", wrong)
    print("Accuracy Percentage", correct/(correct + wrong) * 100)

#group

if reco_type is "group":
    face_cascade =cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    dir = "images/GroupImages/"

    img = cv2.imread(dir + "group.jpg", 0)

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=3)

    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+h]
        scaled = cv2.resize(roi, (img_width, img_height))
        rec_color = (0, 255, 0)
        rec_stroke = 3
        cv2.rectangle(img, (x,y), (x+w, y+h), rec_color, rec_stroke)

        new_cord = my_algo_class_obj.new_cords(scaled)
        name = my_algo_class_obj.recognize_face(new_cord)
        font = cv2.FONT_HERSHEY_COMPLEX
        font_color = (255, 0, 0)
        font_stroke = 3
        cv2.putText(img, name, (x,y), font, 5, font_color, font_stroke, cv2.LINE_AA)

    frame = cv2.resize(img, (1000, 568))
    cv2.imshow("Frame", frame)
    cv2.waitKey()

#video

if reco_type is "video":
    face_cascade =cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=3)

        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+h]
            scaled = cv2.resize(roi, (img_width, img_height))
            rec_color = (0, 255, 0)
            rec_stroke = 3
            cv2.rectangle(img, (x,y), (x+w, y+h), rec_color, rec_stroke)

            new_cord = my_algo_class_obj.new_cords(scaled)
            name = my_algo_class_obj.recognize_face(new_cord)
            font = cv2.FONT_HERSHEY_COMPLEX
            font_color = (255, 0, 0)
            font_stroke = 3
            cv2.putText(img, name, (x,y), font, 1, font_color, font_stroke, cv2.LINE_AA)

        frame = cv2.resize(img, (1000, 568))
        cv2.imshow("Frame", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


