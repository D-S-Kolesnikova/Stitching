import imutils
import cv2
import numpy as np

lowe_ratio = 0.75
max_Threshold=4.0

filename = ["E:/devel/helper/homography/special/2/src2-1.jpg","E:/devel/helper/homography/special/2/src2-2.jpg"]
images = []
num_of_images = np.size(filename)

def Detect_Feature_And_KeyPoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SIFT_create()
    Keypoints = descriptor.detect(image, None)
    (Keypoints, features) = descriptor.compute(image, Keypoints)
    Keypoints = np.float32([i.pt for i in Keypoints])
    return (Keypoints, features)

def image_stitch(images):
    (imageB, imageA) = images
    (KeypointsA, features_of_A) = Detect_Feature_And_KeyPoints(imageA)
    (KeypointsB, features_of_B) = Detect_Feature_And_KeyPoints(imageB)

    match_instance = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_L1)
    All_Matches = match_instance.knnMatch(features_of_A, features_of_B, 2)
    valid_matches = []

    for val in All_Matches:
        if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
            valid_matches.append((val[0].trainIdx, val[0].queryIdx))

    if len(valid_matches) > 4:
        pointsA = np.float32([KeypointsA[i] for (_,i) in valid_matches])
        pointsB = np.float32([KeypointsB[i] for (i,_) in valid_matches])
        (H, status) = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)

    val = imageA.shape[1] + imageB.shape[1]
    result_image = cv2.warpPerspective(imageA, H, (val, imageB.shape[0]))
    cv2.imwrite("Warped_image.jpg",result_image)
    result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    return result_image

####################################
for i in range(num_of_images):
    images.append(cv2.imread(filename[i]))
    
# for i in range(num_of_images):
#     images[i] = imutils.resize(images[i], width=800)

# for i in range(num_of_images):
#     images[i] = imutils.resize(images[i], height=800)

if num_of_images==2:
    result = image_stitch([images[0], images[1]])
else:
    result = image_stitch([images[num_of_images-2], images[num_of_images-1]])
    for i in range(num_of_images - 2):
        result = image_stitch([images[num_of_images-i-3],result])      

cv2.imwrite("Panorama_image.jpg",result)
    
    
