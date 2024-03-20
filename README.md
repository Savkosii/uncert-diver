# Uncertainty Attended Multiscale Integration for Volume Rendering
## Motivation
[DIVeR](https://github.com/lwwu2/diver) suffers from aliasing problem due to the voxel features representation of 3D scene. Structures near the scale of the voxel grid can cause blocky aliasing artifacts. Unlike [Mip-NeRF](https://github.com/google/mipnerf), which alleviate the aliasing problem by softly removing the high frequency component during the integration of positional embedding, we adopt a locally super-sampling approach: find those voxels that are more likely to cause aliasing and explicitly increase their resolutions. Our solution is based on the observation that the area suffered from aliasing artifacts tends to have higher uncertainty in model prediction.

## Method
### Uncertainty Estimation
To esitimate uncertainty, we predict the color of ray segment $c(r(t_i))$ as a Gaussian distribution instead of deterministic point, with variance (uncertainty) $\sigma^2(r(t_i))$ . Following the assumption that the color distribution of different ray segments are i.i.d., the color of rendered ray $c(r)$ is a mixture of Gaussian, with variance $\sum \sigma^2(r(t_i))$. We train the model by maximizing the likelyhood of $c(r)$. 

![uncert_map1](https://github.com/Savkosii/uncert-diver/blob/master/images/uncert_map1.png)

![uncert_map2](https://github.com/Savkosii/uncert-diver/blob/master/images/uncert_map2.png)

### Uncertainty Attended Rendering
At the pruning stage of model, we not only prune the voxels with low occupancy, but also highlight the voxels with high uncertainty (above a configurable threshold), which typically covers high frequency details like edges, and 2x the scale of these voxels.

Following vanilla Diver, we compute the voxel feature of ray as the concatenation of its ray segmentation feature, which, however, can be segmented by voxels at different scales.

## Result
Our method does not introduce much overhead since we only highlight nearly 0.1~0.6% voxels among the whole scene, but achieve a decent improvement over the vaniall Diver on more challenging (i.e., model gain relatively lower PSNR) dataset.

![result](https://pic.imgdb.cn/item/65faec129f345e8d03eb5ccf.png)

## Reproduce
See the configuartion and training procedure of [DiVeR](https://github.com/lwwu2/diver)
