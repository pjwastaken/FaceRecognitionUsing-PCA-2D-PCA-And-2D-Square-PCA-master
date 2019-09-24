import numpy as np
import cv2
import scipy.linalg as s_linalg

class my_algo:
    def __init__(self, image_matrix, image_labels, image_targets, no_of_elements, images_width, images_height, quality_percent):
        self.image_matrix = image_matrix
        self.image_labels = image_labels
        self.image_targets = image_targets
        self.no_of_elements = no_of_elements
        self.images_width = images_width
        self.images_height = images_height
        self.quality_percent = quality_percent

        mean = np.mean(self.image_matrix, 1)
        self.mean_face = np.asmatrix(mean).T
        self.image_matrix -= self.mean_face

    def give_P_value(self, eig_vals):
        sum_original = np.sum(eig_vals)
        sum_threshold = sum_original * self.quality_percent/100
        sum_temp = 0
        P = 0
        while sum_temp < sum_threshold:
            sum_temp += eig_vals[P]
            P += 1
        return P

    def reduce_dim(self):
        u, eig_vals, v_t = s_linalg.svd(self.image_matrix, full_matrices=True)
        P = self.give_P_value(eig_vals)
        self.new_bases = u[:, 0:P]
        self.new_coordinates = np.dot(self.new_bases.T, self.image_matrix)
        return self.new_coordinates

    def new_cords(self, single_image):
        img_vec = np.asmatrix(single_image).ravel()
        img_vec = img_vec.T
        new_mean = ((self.mean_face * len(self.image_labels)) + img_vec)/(len(self.image_labels) + 1)
        img_vec = img_vec - new_mean
        return np.dot(self.new_bases.T, img_vec)

    def recognize_face(self, new_cords_of_image):
        classes = len(self.no_of_elements)
        start = 0
        dist = []
        for i in range(classes):
            temp_imgs = self.new_coordinates[:, int(start):int(start + self.no_of_elements[i])]
            mean_temp = np.asmatrix(np.mean(temp_imgs, 1)).T
            start = start+self.no_of_elements[i]
            dist_temp = np.linalg.norm(new_cords_of_image - mean_temp)
            dist += [dist_temp]

        min_pos = np.argmin(dist)
        return self.image_targets[min_pos]

    def img_from_path(self, path):
        gray = cv2.imread(path, 0)
        return cv2.resize(gray, (self.images_width, self.images_height))

    def new_to_old_cords(self, new_cords):
        return self.mean_face + (np.asmatrix(np.dot(self.new_bases, new_cords))).T

    def show_image(self, label_to_show, old_cords):
        old_cords_matrix = np.reshape(old_cords, [self.images_width, self.images_height])
        old_cords_integers = np.array(old_cords_matrix, dtype=np.uint8)
        resized_image = cv2.resize(old_cords_integers, (500, 500))
        cv2.imshow(label_to_show, resized_image)
        cv2.waitKey()

    def show_eigen_faces(self, min_pix_int, max_pix_int, eig_face_no):
        ev = self.new_bases[:, eig_face_no: eig_face_no + 1]
        min_orig = np.min(ev)
        max_orig = np.max(ev)

        ev = min_pix_int + (((max_pix_int - min_pix_int)/(max_orig - min_orig)) * ev)
        self.show_image("Eigen Face"+str(eig_face_no),ev)





    

