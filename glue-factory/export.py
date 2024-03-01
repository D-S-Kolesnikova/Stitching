import argparse
from typing import List
from pathlib import Path

import torch
import cv2
import numpy as np

from gluefactory.utils.image import ImagePreprocessor, load_image
from kornia.color import rgb_to_grayscale
from gluefactory.models.extractors.superpoint_open import SuperPoint
from gluefactory.models.matchers.lightglue import LightGlue, normalize_keypoints

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_size",
        nargs="+",
        type=int,
        default=512,
        required=False,
        help="Sample image size for ONNX tracing. If a single integer is given, resize the longer side of the image to this value. Otherwise, please provide two integers (height width).",
    )
    parser.add_argument(
        "--extractor_type",
        type=str,
        default="superpoint",
        choices=["superpoint", "disk"],
        required=False,
        help="Type of feature extractor. Supported extractors are 'superpoint' and 'disk'. Defaults to 'superpoint'.",
    )
    parser.add_argument(
        "--extractor_path",
        type=str,
        default=None,
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
    parser.add_argument(
        "--end2end",
        action="store_true",
        help="Whether to export an end-to-end pipeline instead of individual models.",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Whether to allow dynamic image sizes."
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
    img_size=512,
    extractor_path=None,
    lightglue_path=None,
    img0_path="assets/boat1.png",
    img1_path="assets/boat2.png",
    max_num_keypoints=None,
):
    # Handle args
    if isinstance(img_size, List) and len(img_size) == 1:
        img_size = img_size[0]

    # Sample images for tracing
    image0 = load_image(img0_path, resize=(1024, 1024))
    image1 = load_image(img1_path, resize=(1024, 1024))
    # Models
    # SuperPoint works on grayscale images.
    image0 = rgb_to_grayscale(image0)
    image1 = rgb_to_grayscale(image1)
    superpoint_conf = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "max_num_keypoints": 2048,
        "force_num_keypoints": False,
        "detection_threshold": 0.005,
        "remove_borders": 4,
        "descriptor_dim": 256,
        "channels": [64, 64, 128, 128, 256],
        "dense_outputs": None,
    }
    extractor = SuperPoint(superpoint_conf).eval()
    lightglue_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.0,  # match threshold
        "checkpointed": False,
        "weights": "D:\\glue-factory\\outputs\\training\\ckeck_training\\checkpoint_best.pth",  # either a path or the name of pretrained weights (disk, ...)
        "weights_from_version": "v0.1_arxiv",
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        }
    }
    print(Path(lightglue_conf["weights"]).exists())
    lightglue = LightGlue(lightglue_conf).eval()

    # ONNX Export
    
    # # Export Extractor
    # torch.onnx.export(
    #     extractor,
    #     image0[None],
    #     extractor_path,
    #     input_names=["image"],
    #     output_names=["keypoints", "scores", "descriptors"],
    #     opset_version=17,
    # )
    # Export LightGlue
    feats0, feats1 = extractor({"image": image0[None]}), extractor({"image": image1[None]})
    kpts0, scores0, desc0 = feats0["keypoints"], feats0["keypoint_scores"], feats0["descriptors"]
    kpts1, scores1, desc1 = feats1["keypoints"], feats1["keypoint_scores"], feats1["descriptors"]
    data = {
                "keypoints0": kpts0,
                "keypoints1": kpts1,
                "descriptors0" : desc0.detach(),
                "descriptors1" : desc1.detach(),
                "view0" : { "image_size" : torch.tensor([[image0.shape[1], image0.shape[2]]])},
                "view1" : { "image_size" : torch.tensor([[image1.shape[1], image1.shape[2]]])},
            }
    torch.onnx.export(
        lightglue,
        (data, {}),
        lightglue_path,
        input_names=["keypoints0", "keypoints1", "descriptors0", "descriptors1", "view0", "view1"],
        output_names=[
            "matches0",
            "matches1",
            "matching_scores0",
            "matching_scores1",
            "log_assignment"],
        opset_version=17
    )


if __name__ == "__main__":
    args = parse_args()
    export_onnx(
        img_size=512,
        extractor_path="weights/superpoint-open",
        lightglue_path="weights/lightglue-open",
        img0_path="assets/boat1.png",
        img1_path="assets/boat2.png",
        max_num_keypoints=2048)
