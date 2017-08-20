# Introduction to this project

A major advantage of deep learning is that useful features can be learned automatically from images. In Kim & Brunner's previous work, they showed that convolutional neural networks (ConvNets) are able to produce accurate star-galaxy classifications by learning directly from the pixel values of photometric images.

However, when using supervised learning algorithms, we need a lot of manual tagging data or images, which is pretty time/resource consuming. If we can use a unsupervised learning way to extract useful features automaticly and then run clustering algorithms on those extracted features, we may realize unsupervised star-galaxy classification as well as pixel wise classification.

In [Kim's previous work](https://github.com/EdwardJKim/unsupervised-dl4astro/blob/master/notebooks/kmeans.ipynb), he proposed an scheme to train a classifier without any labels: He assigned "surrogate" classes, and train the network to discriminate between a set of surrogate classes. 

The biggest problem lies in the number of surrogate classes. If we use N surrogate classes, we need to use N output units in the last layer. This can cause the network to become extremely complex, also can cause overfitting problems.

Here I try to use a variational auto-encoders(VAEs, for classification) and auto-encoders(AEs, for segmentation) instead of use surrogate classes. 

<img src="https://github.com/tensorstone/Galaxy_Zoo/blob/master/Structurefig.001.jpeg?raw=true" width=850 height=850 />

For the classification task, I hope I can make use of VAEs' dimensionality reduction ability to make clustering easier. And then I can try different clustering algorithms to separate the hidden variables of in the hidden space. However, VAEs are not designed to be a classifier but a generator. In order to get better classification performance, I tried to revise the loss function of VAEs. From both mathematical and practical perspective, VAE with new loss function works better, which means a higher accuracy can be obtained after the cluster step followed by the VAE dim-reduction process.

For the segmentation task, I used AEs instead of VAEs, for we don't expect the neural net to be able to generate new images. The only goal of our neural net here is to reproduce the original images with a denoising effect. To obtain better performance, I tried to use residual connections between each down-sampling layer and the correspond up-sampling layer.

In the classification task, the best result I got was higher than 88.9%(accuracy), using a star-galaxy proportioin priori. And an area under curve(AUC) over 0.92 is onbtained if the groundtruth labels are include. I tried supervised learning to make a comparision and that supervised accuracy is about 95%.

In the unsupervised segmentation task, here is a trade-off problem: if we want to obtain a higher sensitivity that we can detect as many objects as possible, including the faint ones, we must lower the neural net's detection threshold. But then some of the noise pixels will also be detected as stars or galaxies. In pursuit of better results, pixel level labeled data must be introduced. Sofie and I will continue to work on this task.

We can then combine the two networks above. A network has the ability to perform unsupervised star-galaxy segementation and classification can be obtained.

The following picture shows part of the segmentation and classification result. The neural net found 38 of 40 central objects, and the classification accuracy is about 89%.
<img src="http://a2.qpic.cn/psb?/V11SDUzR1bBoP8/j*chlG2HF9DqMBDGdY.oYcM05Xxm1bSMZ.zMqI7eE8U!/b/dGkBAAAAAAAA&bo=agQ4BHEEPwQDCTA!&rf=viewer_4" width=450 height=450 />
$$Figure1.\quad Segmentation\quad result$$

<img src="http://a1.qpic.cn/psb?/V11SDUzR1bBoP8/YkyjVI2hV1rkeAt3ZS6qFmHRixgMFc4eGiBezaPUV2w!/b/dD4BAAAAAAAA&bo=cQQgAnEEIAIDCSw!&rf=viewer_4" width=450 height=450 />
$$Figure2.\quad Segmentation\quad and \quad classification$$
