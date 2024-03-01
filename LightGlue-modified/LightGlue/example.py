from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
from lightglue import match_pair
import cv2

# SuperPoint+LightGlue
# extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
# matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
name1=("E:/devel/helper/homography/results/src5-1.jpg")
name2=("E:/devel/helper/homography/results/src5-2.jpg")

image0 = load_image(name1).cuda()
image1 = load_image(name2).cuda()

feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)
kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

kptsB = m_kpts0.cpu().numpy()
kptsA = m_kpts1.cpu().numpy()

im_src = cv2.imread(name1)
im_dst = cv2.imread(name2)

(H, status) = cv2.findHomography(kptsA, kptsB, cv2.RANSAC, 4.0)
val = im_dst.shape[1] + im_src.shape[1]
lon = im_dst.shape[0] + im_src.shape[0]
result_image = cv2.warpPerspective(im_dst, H, (round(val ), round(lon )))
cv2.imwrite("Warped_image.jpg",result_image)
result_image[0:im_src.shape[0], 0:im_src.shape[1]] = im_src
cv2.imwrite("LightGlue.jpg",result_image)