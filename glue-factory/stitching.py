import torch
from


IS = ImageStitcher(KF.LoFTR(pretrained='outdoor'), estimator='ransac').cuda()
    # Compute the stitched result with less GPU memory cost.
    with torch.inference_mode():
    out = IS(img_left, img_right)
    # Show the result
    plt.imshow(K.tensor_to_image(out))