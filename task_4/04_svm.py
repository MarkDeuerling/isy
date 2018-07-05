import numpy as np
import cv2
import glob
from sklearn import svm


############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################

def create_keypoints(w, h):
    '''
    This function creates a  grid of key points for a specific image size, e.g. 256x256
    :param w: width of the grid
    :param h: height of the grid
    :return: list of key points
    '''
    kps = []
    kps_size = 15

    for x in range(0, w, kps_size):
        for y in range(0, h, kps_size):
            kp = cv2.KeyPoint(x, y, kps_size)
            kps.append(kp)

    return kps


def gen_feature(img, img_kp):
    img = cv2.imread(img)
    img_des = sift.compute(img, img_kp)[1]
    return img_des.flatten()


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px
images_train_car = glob.glob('images/db/train/cars/*.jpg')
images_train_face = glob.glob('images/db/train/faces/*.jpg')
images_train_flower = glob.glob('images/db/train/flowers/*.jpg')

sift = cv2.xfeatures2d.SIFT_create()  # create sift
img_kp = create_keypoints(256, 256)

train_X = []
zipped = np.concatenate([images_train_car, images_train_face, images_train_flower]).flatten()

for img in zipped:
    train_X.append(gen_feature(img, img_kp))

car_labels = np.repeat(0, len(images_train_car)).tolist()
face_labels = np.repeat(1, len(images_train_face)).tolist()
flower_labels = np.repeat(2, len(images_train_flower)).tolist()
train_Y = np.append(car_labels, face_labels)
train_Y = np.append(train_Y, flower_labels)
print(train_Y)


# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers

# 3. We use scikit-learn to train a SVM classifier. Specifically we use a LinearSVC in our case. Have a look at the documentation.
# You will need .fit(X_train, y_train)
lin_clf = svm.LinearSVC()
print("train_x len:", len(train_X))
print("train_y len:", len(train_Y))
lin_clf.fit(train_X, train_Y)


# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image
images_test = glob.glob('images/db/test/*.jpg')

classes = {
    0: 'car',
    1: 'face',
    2: 'flower'
}

for img in images_test:
    des = gen_feature(img, img_kp).reshape(1, -1)
    print(des)
    predicted_class = lin_clf.predict(des)[0]

    img_class = classes.get(predicted_class)

    img = cv2.imread(img)
    cv2.imshow(img_class, img)  # show video stream
    keyInput = cv2.waitKey(0)  # wait one millisecond for key input by user
    if keyInput == ord('q'):
        continueStream = False
    cv2.destroyAllWindows()
# 5. output the class + corresponding name