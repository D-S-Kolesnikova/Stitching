
import argparse
from typing import List

import torch

from utils import load_image, rgb_to_grayscale
from superpoint_pytorch import SuperPoint

from end2end import LightGlueEnd2End
from lightglue import LightGlue
from super_lightglue import SuperPointForLightglue

import kornia


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        default=1024,
        required=False,
        help="Sample image size for ONNX tracing. If a single integer is given, resize the longer side of the image to this value. Otherwise, please provide two integers (height width).",
    )
    parser.add_argument(
        "--extractor_path",
        type=str,
        default="weights/superpoint-1024.onnx",
        required=False,
        help="Path to save the feature extractor ONNX model.",
    )
    parser.add_argument(
        "--lightglue_path",
        type=str,
        default=None,
        required=False,
        help="Path to save the LightGlue ONNX model.",
    )
    # Extractor-specific args:
    parser.add_argument(
        "--max_num_keypoints",
        type=int,
        default=None,
        required=False,
        help="Maximum number of keypoints outputted by the extractor.",
    )

    return parser.parse_args()

def export_onnx(
    img_size=[512,512],
    extractor_path=None,
    lightglue_path=None,
    img0_path=("E:/devel/helper/homography/special/2/src2-1.jpg"),
    img1_path=("E:/devel/helper/homography/special/2/src2-2.jpg"),
    max_num_keypoints=2048,
):
        
    # Sample images for tracing
    image0, scales0 = load_image(img0_path, resize=img_size)
    image1, scales1 = load_image(img1_path, resize=img_size)
    
    image0 = kornia.geometry.transform.resize(
                image0, img_size, side='long',
                antialias=True,
                align_corners=None)
    image1 = kornia.geometry.transform.resize(
                image1, img_size, side='long',
                antialias=True,
                align_corners=None)
    
    image0 = rgb_to_grayscale(image0)
    image1 = rgb_to_grayscale(image1)
    dummy_input = torch.randn(1, 1, 512, 512)
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval()
    with torch.autocast("cuda"):
            torch.onnx.export(
                extractor,
                image0[None],
                extractor_path,
                input_names=["image"],
                output_names=["keypoints", "scores", "descriptors"],
                opset_version=17
            )

def export_onnx_end2end(
    img_size=[512,512],
    extractor_path=None,
    lightglue_path=None,
    img0_path=("E:/devel/helper/homography/special/2/src2-1.jpg"),
    img1_path=("E:/devel/helper/homography/special/2/src2-2.jpg"),
    max_num_keypoints=2048,
):
        
    # Sample images for tracing
    image0, scales0 = load_image(img0_path, resize=img_size)
    image1, scales1 = load_image(img1_path, resize=img_size)
    
    image0 = kornia.geometry.transform.resize(
                image0, img_size, side='long',
                antialias=True,
                align_corners=None)
    image1 = kornia.geometry.transform.resize(
                image1, img_size, side='long',
                antialias=True,
                align_corners=None)
    
    image0 = rgb_to_grayscale(image0)
    image1 = rgb_to_grayscale(image1)
    dummy_input = torch.randn(1, 1, 512, 512)
    
    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval()
    lightglue = LightGlue("superpoint").eval()
    pipeline = LightGlueEnd2End(extractor, lightglue).eval()
    
    with torch.autocast("cuda"):
            torch.onnx.export(
                pipeline,
                (image0[None], image1[None]),
                extractor_path,
                input_names=["image0", "image1"],
                output_names=[                
                    "kpts0",
                    "kpts1",
                    "matches0",
                    "mscores0",
                ],
                opset_version=17
            )

def export_onnx_forLightGlue(
    img_size=[512,512],
    extractor_path=None,
    lightglue_path=None,
    img0_path=("E:/devel/helper/homography/special/2/src2-1.jpg"),
    img1_path=("E:/devel/helper/homography/special/2/src2-2.jpg"),
    max_num_keypoints=2048,
):
        
     # Sample images for tracing
    image0, scales0 = load_image(img0_path)
    image1, scales1 = load_image(img1_path)
    
    image0 = kornia.geometry.transform.resize(
                image0, img_size, side='long',
                antialias=True,
                align_corners=None)
    image1 = kornia.geometry.transform.resize(
                image1, img_size, side='long',
                antialias=True,
                align_corners=None)
    
    image0 = rgb_to_grayscale(image0)
    image1 = rgb_to_grayscale(image1)
    dummy_input = torch.stack((image0, image1))
    extractor = SuperPointForLightglue(max_num_keypoints=max_num_keypoints).eval()
    with torch.autocast("cuda"):
            torch.onnx.export(
                extractor,
                dummy_input,
                extractor_path,
                input_names=["image"],
                output_names=["keypoints0", "keypoints1", "descriptors0", "descriptors1"],
                opset_version=16
            )

if __name__ == "__main__":
    args = parse_args()
    export_onnx(
        [1024,1024],
        extractor_path="weights/superpoint.onnx",
        lightglue_path=None,
        img0_path=("E:/devel/helper/homography/special/2/src2-1.jpg"),
        img1_path=("E:/devel/helper/homography/special/2/src2-2.jpg"),
        max_num_keypoints=8192)