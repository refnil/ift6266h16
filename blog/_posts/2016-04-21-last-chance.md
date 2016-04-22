---
title: It didn't work out...
---

So, I've made my last try today to train a neural net for the project. 

It had three convolutionnal layer with 100, 200 and 300 filters at each layers respectively. All used 5 by 5 convolution with rectifier as an activation function. They were followed with  2 by 2 pooling layers. It was then fully connected with a 500 units layer before doing the classification.

The neural net was trainned with dropout, L1 and L2 regularisation, RMSProp and data augmentation. The image were resized to 128 by 128.

It did quite worse than I taught it would, peaking at 52% test accuracy. 
