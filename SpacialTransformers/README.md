## Spatial Transformer

You can read more about the spatial transformer networks in the [DeepMind paper](https://arxiv.org/abs/1506.02025)

Spatial transformer networks are a generalization of differentiable attention to any spatial transformation. Spatial transformer networks (STN for short) allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model. For example, it can crop a region of interest, scale and correct the orientation of an image. It can be a useful mechanism because CNNs are not invariant to rotation and scale and more general affine transformations. One of the best things about STN is the ability to simply plug it into any existing CNN with very little modification

Below animation shows the rotated MNIST:

![](D:\Learnings\github\EVA6_Assignments_Session12\SpacialTransformers\Images\1 P_nv_a_Q3LqM9d10XGE3TQ.gif)



