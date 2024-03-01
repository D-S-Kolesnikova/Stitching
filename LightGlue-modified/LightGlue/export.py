from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import ImagePreprocessor
import cv2
import torch
import kornia

from lightglue.utils import load_image, rbd, batch_to_device

def export_onnx(
    img_size,
    extractor_path,
    lightglue_path,
    img0_path,
    img1_path,
    max_num_keypoints=2048,
):
        
    # Sample images for tracing
    img0 = load_image(img0_path).cuda()
    img1 = load_image(img1_path).cuda()
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda() 
    lightglue = LightGlue(features='superpoint').eval().cuda()
    # extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(device="cuda", dtype=torch.float32)
    # if img.dim() == 3:
    #         img = img[None]  # add batch dim
    # img, scales = ImagePreprocessor(**{**extractor.preprocess_conf})(img)
    # feats = extractor.forward({"image": img})
    # with torch.autocast("cuda"):
    #         torch.onnx.export(
    #             extractor,
    #             img0[None],
    #             extractor_path,
    #             input_names=["image"],
    #             output_names=["keypoints", "descriptors", "keypoint_scores"],
    #             opset_version=16,
    #             verbose=True
    #         )
    feats0, feats1 = extractor({"image": img0[None]}), extractor({"image": img1[None]})
    kpts0, scores0, desc0 = feats0["keypoints"], feats0["keypoint_scores"], feats0["descriptors"]
    kpts1, scores1, desc1 = feats1["keypoints"], feats1["keypoint_scores"], feats1["descriptors"]
    size0, size1 = torch.tensor([[img0.shape[1], img0.shape[2]]]), torch.tensor([[img1.shape[1], img1.shape[2]]])
    # torch.onnx.export(
    #     lightglue,
    #     ({"image0": feats0, "image1": feats1}, {}),
    #     lightglue_path,
    #     output_names=[
    #         "matches0",
    #         "scores0"],
    #     opset_version=17
    # )
    matches01 = lightglue({"image0": feats0, "image1": feats1})
    data = [feats0, feats1, matches01]
    feats0, feats1, matches01 = [batch_to_device(rbd(x), "cpu") for x in data]
    matches = matches01['matches']
    m_kpts0, m_kpts1 = kpts0[0, matches[..., 0]], kpts1[0, matches[..., 1]]
    kptsB = m_kpts0.cpu().numpy()
    kptsA = m_kpts1.cpu().numpy()
    im_src = cv2.imread(img0_path)
    im_dst = cv2.imread(img1_path)
    (H, status) = cv2.findHomography(kptsA, kptsB, cv2.RANSAC, 4.0)
    val = im_dst.shape[1] + im_src.shape[1]
    lon = im_dst.shape[0] + im_src.shape[0]
    result_image = cv2.warpPerspective(im_dst, H, (round(val ), round(lon )))
    cv2.imwrite("Warped_image.jpg",result_image)
    result_image[0:im_src.shape[0], 0:im_src.shape[1]] = im_src
    cv2.imwrite("LightGlue.jpg",result_image)
    
if __name__ == "__main__":
    export_onnx(
        [1024,1024],
        extractor_path="E:/devel/helper/homography/projects/LightGlue-new/LightGlue/weights/superpoint.onnx",
        lightglue_path="E:/devel/helper/homography/projects/LightGlue-new/LightGlue/weights/lightglue_new.onnx",
        img0_path=("E:/devel/helper/homography/results/src5-1.jpg"),
        img1_path=("E:/devel/helper/homography/results/src5-2.jpg"),
        max_num_keypoints=2048)