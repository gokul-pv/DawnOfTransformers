## Spatial Transformer

> You can read more about the spatial transformer networks in the [DeepMind paper](https://arxiv.org/abs/1506.02025)
>

Spatial transformer networks are a generalization of differentiable attention to any spatial transformation. Spatial transformer networks (STN for short) allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model. For example, it can crop a region of interest, scale and correct the orientation of an image. It can be a useful mechanism because CNNs are not invariant to rotation and scale and more general affine transformations. One of the best things about STN is the ability to simply plug it into any existing CNN with very little modification

There are mainly 3 transformation learnt by STN in the DeepMind paper

- **Affine Transformation** - An affine transformation is any transformation that preserves collinearity (i.e., all points lying on a line initially still lie on a line after transformation) and ratios of distances (e.g., the midpoint of a line segment remains the midpoint after transformation).
- **Projective Transformation** - A projective transformation is a transformation used in projective geometry:  It describes what happens to the perceived positions of observed  objects when the point of view of the observer changes. Projective  transformations do not preserve sizes or angles but do preserve  incidence and cross-ratio: two properties which are important in  projective geometry. 
- **Thin Plate Spline (TPS) Transformation** - Thin plate splines (TPS) are a spline-based technique for data interpolation and smoothing

Below animation shows the rotated MNIST:

![Figure 1](https://github.com/gokul-pv/DawnOfTransformers/blob/main/SpacialTransformers/Images/1%20P_nv_a_Q3LqM9d10XGE3TQ.gif)



## Depicting spatial transformer networks

Spatial transformer networks boils down to three main components :

- The localization network is a regular CNN which regresses the transformation parameters. The transformation is never learned explicitly from this dataset, instead the network learns automatically the spatial transformations that enhances the global accuracy.
- The grid generator generates a grid of coordinates in the input image corresponding to each pixel from the output image.
- The sampler uses the parameters of the transformation and applies it to the input image.

![](https://github.com/gokul-pv/DawnOfTransformers/blob/main/SpacialTransformers/Images/stn-arch.png)



### Localization Net

With **input feature map \*U\***, with width *W*, height *H* and *C* channels, **outputs are \*θ\***, the parameters of transformation *Tθ*. It can be learnt as affine transform as above. Or to be more  constrained such as the used for attention which only contains scaling and translation.

### Grid Generator

- Suppose we have a regular grid G, this G is a set of points with **target coordinates (xt_i, yt_i)**.
- Then we **apply transformation *Tθ* on G**, i.e. *Tθ*(*G*).
- After *Tθ*(*G*), a set of points with **destination coordinates (xt_i, yt_i) is outputted**. These points have been altered based on the transformation parameters.  It can be Translation, Scale, Rotation or More Generic Warping depending on how we set *θ* as mentioned above.

### Sampler

- Based on the new set of coordinates (xt_i, yt_i), we generate a transformed output feature map *V*. This *V* is translated, scaled, rotated, warped, projective transformed or affined, whatever.
- It is noted that STN can be applied to not only input image, but also intermediate feature maps.



## Links to Code

Github Link - https://github.com/gokul-pv/DawnOfTransformers/blob/main/SpacialTransformers/SpatialTransformer.ipynb

Colab Link - https://colab.research.google.com/github/gokul-pv/DawnOfTransformers/blob/main/SpacialTransformers/SpatialTransformer.ipynb



## Model Architecture

```
Requirement already satisfied: torchsummary in /usr/local/lib/python3.7/dist-packages (1.5.1)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 26, 26]           9,472
         MaxPool2d-2           [-1, 64, 13, 13]               0
              ReLU-3           [-1, 64, 13, 13]               0
            Conv2d-4            [-1, 128, 9, 9]         204,928
         MaxPool2d-5            [-1, 128, 4, 4]               0
              ReLU-6            [-1, 128, 4, 4]               0
            Linear-7                  [-1, 256]         524,544
              ReLU-8                  [-1, 256]               0
            Linear-9                    [-1, 6]           1,542
           Conv2d-10           [-1, 16, 28, 28]           1,216
           Conv2d-11           [-1, 32, 10, 10]          12,832
        Dropout2d-12           [-1, 32, 10, 10]               0
           Linear-13                 [-1, 1024]         820,224
           Linear-14                   [-1, 10]          10,250
================================================================
Total params: 1,585,008
Trainable params: 1,585,008
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.76
Params size (MB): 6.05
Estimated Total Size (MB): 6.82
----------------------------------------------------------------
```

## 

## Training and Validation Log

```
Train Epoch: 1 [0/50000 (0%)]	Loss: 2.314888
Train Epoch: 1 [32000/50000 (64%)]	Loss: 2.191543

Test set: Average loss: 1.8399, Accuracy: 3439/10000 (34%)

Train Epoch: 2 [0/50000 (0%)]	Loss: 1.908740
Train Epoch: 2 [32000/50000 (64%)]	Loss: 1.924799

Test set: Average loss: 1.6107, Accuracy: 4266/10000 (43%)

Train Epoch: 3 [0/50000 (0%)]	Loss: 1.644960
Train Epoch: 3 [32000/50000 (64%)]	Loss: 1.494569

Test set: Average loss: 1.5346, Accuracy: 4567/10000 (46%)

Train Epoch: 4 [0/50000 (0%)]	Loss: 1.810462
Train Epoch: 4 [32000/50000 (64%)]	Loss: 1.585784

Test set: Average loss: 1.4575, Accuracy: 4790/10000 (48%)

Train Epoch: 5 [0/50000 (0%)]	Loss: 1.581916
Train Epoch: 5 [32000/50000 (64%)]	Loss: 1.264176

Test set: Average loss: 1.3822, Accuracy: 5053/10000 (51%)

Train Epoch: 6 [0/50000 (0%)]	Loss: 1.639581
Train Epoch: 6 [32000/50000 (64%)]	Loss: 1.378260

Test set: Average loss: 1.3566, Accuracy: 5157/10000 (52%)

Train Epoch: 7 [0/50000 (0%)]	Loss: 1.205654
Train Epoch: 7 [32000/50000 (64%)]	Loss: 1.391012

Test set: Average loss: 1.3055, Accuracy: 5332/10000 (53%)

Train Epoch: 8 [0/50000 (0%)]	Loss: 1.698826
Train Epoch: 8 [32000/50000 (64%)]	Loss: 1.572415

Test set: Average loss: 1.3065, Accuracy: 5392/10000 (54%)

Train Epoch: 9 [0/50000 (0%)]	Loss: 1.254889
Train Epoch: 9 [32000/50000 (64%)]	Loss: 1.218326

Test set: Average loss: 1.2680, Accuracy: 5550/10000 (56%)

Train Epoch: 10 [0/50000 (0%)]	Loss: 1.641240
Train Epoch: 10 [32000/50000 (64%)]	Loss: 1.394785

Test set: Average loss: 1.2118, Accuracy: 5854/10000 (59%)

Train Epoch: 11 [0/50000 (0%)]	Loss: 1.258880
Train Epoch: 11 [32000/50000 (64%)]	Loss: 1.183975

Test set: Average loss: 1.2484, Accuracy: 5591/10000 (56%)

Train Epoch: 12 [0/50000 (0%)]	Loss: 1.310694
Train Epoch: 12 [32000/50000 (64%)]	Loss: 1.401218

Test set: Average loss: 1.1827, Accuracy: 5873/10000 (59%)

Train Epoch: 13 [0/50000 (0%)]	Loss: 1.277470
Train Epoch: 13 [32000/50000 (64%)]	Loss: 1.288276

Test set: Average loss: 1.2404, Accuracy: 5690/10000 (57%)

Train Epoch: 14 [0/50000 (0%)]	Loss: 1.182854
Train Epoch: 14 [32000/50000 (64%)]	Loss: 1.512306

Test set: Average loss: 1.1728, Accuracy: 5980/10000 (60%)

Train Epoch: 15 [0/50000 (0%)]	Loss: 1.289078
Train Epoch: 15 [32000/50000 (64%)]	Loss: 1.366868

Test set: Average loss: 1.1327, Accuracy: 6220/10000 (62%)

Train Epoch: 16 [0/50000 (0%)]	Loss: 1.163321
Train Epoch: 16 [32000/50000 (64%)]	Loss: 1.170165

Test set: Average loss: 1.1215, Accuracy: 6205/10000 (62%)

Train Epoch: 17 [0/50000 (0%)]	Loss: 1.281437
Train Epoch: 17 [32000/50000 (64%)]	Loss: 1.338845

Test set: Average loss: 1.1141, Accuracy: 6148/10000 (61%)

Train Epoch: 18 [0/50000 (0%)]	Loss: 1.138275
Train Epoch: 18 [32000/50000 (64%)]	Loss: 1.190396

Test set: Average loss: 1.0954, Accuracy: 6283/10000 (63%)

Train Epoch: 19 [0/50000 (0%)]	Loss: 1.152543
Train Epoch: 19 [32000/50000 (64%)]	Loss: 1.276010

Test set: Average loss: 1.0616, Accuracy: 6403/10000 (64%)

Train Epoch: 20 [0/50000 (0%)]	Loss: 1.271586
Train Epoch: 20 [32000/50000 (64%)]	Loss: 1.059713

Test set: Average loss: 1.1060, Accuracy: 6198/10000 (62%)

Train Epoch: 21 [0/50000 (0%)]	Loss: 1.178607
Train Epoch: 21 [32000/50000 (64%)]	Loss: 0.998532

Test set: Average loss: 1.0670, Accuracy: 6358/10000 (64%)

Train Epoch: 22 [0/50000 (0%)]	Loss: 1.002847
Train Epoch: 22 [32000/50000 (64%)]	Loss: 0.776997

Test set: Average loss: 1.0880, Accuracy: 6261/10000 (63%)

Train Epoch: 23 [0/50000 (0%)]	Loss: 1.323140
Train Epoch: 23 [32000/50000 (64%)]	Loss: 0.977235

Test set: Average loss: 1.0894, Accuracy: 6279/10000 (63%)

Train Epoch: 24 [0/50000 (0%)]	Loss: 1.155735
Train Epoch: 24 [32000/50000 (64%)]	Loss: 0.903124

Test set: Average loss: 1.0578, Accuracy: 6380/10000 (64%)

Train Epoch: 25 [0/50000 (0%)]	Loss: 1.076518
Train Epoch: 25 [32000/50000 (64%)]	Loss: 1.064416

Test set: Average loss: 1.0651, Accuracy: 6385/10000 (64%)

Train Epoch: 26 [0/50000 (0%)]	Loss: 0.859532
Train Epoch: 26 [32000/50000 (64%)]	Loss: 0.819383

Test set: Average loss: 1.0631, Accuracy: 6394/10000 (64%)

Train Epoch: 27 [0/50000 (0%)]	Loss: 0.805778
Train Epoch: 27 [32000/50000 (64%)]	Loss: 0.878193

Test set: Average loss: 1.0095, Accuracy: 6589/10000 (66%)

Train Epoch: 28 [0/50000 (0%)]	Loss: 0.752389
Train Epoch: 28 [32000/50000 (64%)]	Loss: 0.919929

Test set: Average loss: 1.0278, Accuracy: 6490/10000 (65%)

Train Epoch: 29 [0/50000 (0%)]	Loss: 1.058492
Train Epoch: 29 [32000/50000 (64%)]	Loss: 0.705471

Test set: Average loss: 1.1034, Accuracy: 6216/10000 (62%)

Train Epoch: 30 [0/50000 (0%)]	Loss: 0.865330
Train Epoch: 30 [32000/50000 (64%)]	Loss: 0.805255

Test set: Average loss: 1.0153, Accuracy: 6535/10000 (65%)

Train Epoch: 31 [0/50000 (0%)]	Loss: 0.825948
Train Epoch: 31 [32000/50000 (64%)]	Loss: 0.799896

Test set: Average loss: 1.0990, Accuracy: 6298/10000 (63%)

Train Epoch: 32 [0/50000 (0%)]	Loss: 0.941206
Train Epoch: 32 [32000/50000 (64%)]	Loss: 0.942857

Test set: Average loss: 1.1288, Accuracy: 6127/10000 (61%)

Train Epoch: 33 [0/50000 (0%)]	Loss: 0.879409
Train Epoch: 33 [32000/50000 (64%)]	Loss: 0.972959

Test set: Average loss: 1.0145, Accuracy: 6566/10000 (66%)

Train Epoch: 34 [0/50000 (0%)]	Loss: 0.963129
Train Epoch: 34 [32000/50000 (64%)]	Loss: 0.877001

Test set: Average loss: 1.0232, Accuracy: 6522/10000 (65%)

Train Epoch: 35 [0/50000 (0%)]	Loss: 0.862200
Train Epoch: 35 [32000/50000 (64%)]	Loss: 0.718072

Test set: Average loss: 1.1563, Accuracy: 6130/10000 (61%)

Train Epoch: 36 [0/50000 (0%)]	Loss: 1.135670
Train Epoch: 36 [32000/50000 (64%)]	Loss: 0.883654

Test set: Average loss: 1.1330, Accuracy: 6141/10000 (61%)

Train Epoch: 37 [0/50000 (0%)]	Loss: 1.106411
Train Epoch: 37 [32000/50000 (64%)]	Loss: 0.877574

Test set: Average loss: 1.0540, Accuracy: 6424/10000 (64%)

Train Epoch: 38 [0/50000 (0%)]	Loss: 0.967103
Train Epoch: 38 [32000/50000 (64%)]	Loss: 0.883978

Test set: Average loss: 0.9957, Accuracy: 6624/10000 (66%)

Train Epoch: 39 [0/50000 (0%)]	Loss: 0.593896
Train Epoch: 39 [32000/50000 (64%)]	Loss: 0.767337

Test set: Average loss: 1.0481, Accuracy: 6507/10000 (65%)

Train Epoch: 40 [0/50000 (0%)]	Loss: 0.787883
Train Epoch: 40 [32000/50000 (64%)]	Loss: 0.819373

Test set: Average loss: 1.0257, Accuracy: 6579/10000 (66%)

Train Epoch: 41 [0/50000 (0%)]	Loss: 0.914633
Train Epoch: 41 [32000/50000 (64%)]	Loss: 0.710645

Test set: Average loss: 1.0936, Accuracy: 6246/10000 (62%)

Train Epoch: 42 [0/50000 (0%)]	Loss: 1.234455
Train Epoch: 42 [32000/50000 (64%)]	Loss: 0.629992

Test set: Average loss: 1.0467, Accuracy: 6487/10000 (65%)

Train Epoch: 43 [0/50000 (0%)]	Loss: 0.632460
Train Epoch: 43 [32000/50000 (64%)]	Loss: 0.704451

Test set: Average loss: 1.0253, Accuracy: 6565/10000 (66%)

Train Epoch: 44 [0/50000 (0%)]	Loss: 0.522004
Train Epoch: 44 [32000/50000 (64%)]	Loss: 0.588414

Test set: Average loss: 1.0218, Accuracy: 6616/10000 (66%)

Train Epoch: 45 [0/50000 (0%)]	Loss: 0.561831
Train Epoch: 45 [32000/50000 (64%)]	Loss: 0.814662

Test set: Average loss: 1.0332, Accuracy: 6532/10000 (65%)

Train Epoch: 46 [0/50000 (0%)]	Loss: 0.685761
Train Epoch: 46 [32000/50000 (64%)]	Loss: 0.646217

Test set: Average loss: 1.0669, Accuracy: 6424/10000 (64%)

Train Epoch: 47 [0/50000 (0%)]	Loss: 1.079817
Train Epoch: 47 [32000/50000 (64%)]	Loss: 0.578733

Test set: Average loss: 1.0877, Accuracy: 6403/10000 (64%)

Train Epoch: 48 [0/50000 (0%)]	Loss: 0.658163
Train Epoch: 48 [32000/50000 (64%)]	Loss: 0.775613

Test set: Average loss: 1.0381, Accuracy: 6565/10000 (66%)

Train Epoch: 49 [0/50000 (0%)]	Loss: 0.524476
Train Epoch: 49 [32000/50000 (64%)]	Loss: 0.694440

Test set: Average loss: 1.0106, Accuracy: 6668/10000 (67%)

Train Epoch: 50 [0/50000 (0%)]	Loss: 0.755872
Train Epoch: 50 [32000/50000 (64%)]	Loss: 0.609947

Test set: Average loss: 1.0224, Accuracy: 6568/10000 (66%)
```



## Visualizing the STN results

![Figure](https://github.com/gokul-pv/DawnOfTransformers/blob/main/SpacialTransformers/Images/stn-result.png)



We can see from the above image that the Spatial Transformer Network has cropped and resized most of the  images to the center. It has rotated many of the images to an orientation that it feels will be helpful.

## Reference

https://arxiv.org/pdf/1506.02025v3.pdf 

https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html https://kevinzakka.github.io/2017/01/10/stn-part1/ https://kevinzakka.github.io/2017/01/18/stn-part2/ https://medium.com/@kushagrabh13/spatial-transformer-networks-ebc3cc1da52d

https://towardsdatascience.com/review-stn-spatial-transformer-network-image-classification-d3cbd98a70aa





