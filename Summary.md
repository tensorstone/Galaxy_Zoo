# Introduction to this project

A major advantage of deep learning is that useful features can be learned automatically from images. In Kim & Brunner's previous work, they showed that convolutional neural networks (ConvNets) are able to produce accurate star-galaxy classifications by learning directly from the pixel values of photometric images.

However, when using supervised learning algorithms, we need a lot of manual tagging data or images, which is pretty time/resource consuming. If we can use a unsupervised learning way to extract useful features automaticly and then run clustering algorithms on those extracted features, we may realize unsupervised star-galaxy classification as well as pixel wise classification.

In [Kim's previous work](https://github.com/EdwardJKim/unsupervised-dl4astro/blob/master/notebooks/kmeans.ipynb), he proposed an scheme to train a classifier without any labels: He assigned "surrogate" classes, and train the network to discriminate between a set of surrogate classes. 

The biggest problem lies in the number of surrogate classes. If we use N surrogate classes, we need to use N output units in the last layer. This can cause the network to become extremely complex, also can cause overfitting problems.

Here I try to use a variational auto-encoders(VAEs, for classification) and auto-encoders(AEs, for segmentation) instead of use surrogate classes. 

For the classification task, I hope I can make use of VAEs' dimensionality reduction ability to make clustering easier. And then I can try different clustering algorithms to separate the hidden variables of in the hidden space. However, VAEs are not designed to be a classifier but a generator. In order to get better classification performance, I tried to revise the loss function of VAEs. From both mathematical and practical perspective, VAE with new loss function works better, which means a higher accuracy can be obtained after the cluster step followed by the VAE dim-reduction process.

For the segmentation task, I used AEs instead of VAEs, for we don't expect the neural net to be able to generate new images. The only goal of our neural net here is to reproduce the original images with a denoising effect. To obtain better performance, I tried to use residual connections between each down-sampling layer and the correspond up-sampling layer.

In the classification task, the best result I got was higher than 88.9%(accuracy), using a star-galaxy proportioin priori. And an area under curve(AUC) over 0.92 is onbtained if the groundtruth labels are include. I tried supervised learning to make a comparision and that supervised accuracy is about 95%.

In the unsupervised segmentation task, here is a trade-off problem: if we want to obtain a higher sensitivity that we can detect as many objects as possible, including the faint ones, we must lower the neural net's detection threshold. But then some of the noise pixels will also be detected as stars or galaxies. In pursuit of better results, pixel level labeled data must be introduced. Sofie and I will continue to work on this task.

We can then combine the two networks above. A network has the ability to perform unsupervised star-galaxy segementation and classification can be obtained.

The following picture shows part of the segmentation and classification result. The neural net found 38 of 40 central objects, and the classification accuracy is about 89%.
<img src="https://github.com/tensorstone/markdownfigs/blob/master/segmentation.png?raw=true" width=450 height=450 />
$$Figure1.\quad Segmentation\quad result$$

<img src="https://github.com/tensorstone/markdownfigs/blob/master/classification.png?raw=true" width=450 height=450 />
$$Figure2.\quad Segmentation\quad and \quad classification$$


# Data set 
## Introduction

In this project, I used the SDSS data set. As a demostration in this notebook, I randomly picked 14100 images and separate them into training set(12000), validation set(2100), with each image has the size of $64\times 64\times 5$ (here are 5 channels: u,g,r,i,z).

I didn't set a test dataset, for in unsupervised learning, we don't use groundtruth labels to train our models, and there is no overfitting possibilities.

## Data preprocessing
Z-score scaling and standard normalization method are introduced in the data preprocessing step. The choose of output activation function should take the pixel value distribution into consideration.


# Part1. Unsupervised Star-Galaxy Classification
## 1.1 The mathematical principles of AEs/VAEs
### 1.1.1 An introduction to AEs and VAEs
In Auto-encoder(AE), we map the input images to a low dimensional hidden space, like the V layer in the following figure:
<img src="https://github.com/tensorstone/markdownfigs/blob/master/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-08-19%2019.08.09.png?raw=true" width=450 height=450 />

The loss function of AE measures the pixel-wise difference between input images and output images. We may call this self-supervised learning. By performing down-sample and upsample process, hot pixels will be removed.

However, if we want to use a given V to generate a image, the result may be bad. Because here the V space is not that continuous/smooth.

To solve this problem(to generate new samples ), we may use a variational auto-encoder(VAE). VAE is always used as a generative model to generate new samples.

For the Encoder part of the VAEs, a certain class of input images are mapped to a certain Multi-dimensional (depends on the number of hidden variables) Gaussian distribution. And then the Decoder use a resampled hidden value to generate a new image.
<img src="https://github.com/tensorstone/markdownfigs/blob/master/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-08-19%2019.08.40.png?raw=true" width=450 height=450 />

<img src="https://github.com/tensorstone/markdownfigs/blob/master/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-08-19%2019.09.31.png?raw=true" width=450 height=450 />


So, the first part of loss function here is also the same: pixel-wise difference between input images and output images.
\begin{align*}
Cross Entropy Loss(p,q) =&-\sum_{x\in \chi} p(x)\log q(x) \\
=& - [y\log h_{\theta}(x) + (1-y)\log(1-h_{\theta}(x))]
\end{align*}


Then, for better performance in generation tasks, we hope that the total distribution can also obey a normal distribution. We can use KL-divergence to describe the difference between two distributions:

$$D_{KL}(P,Q) = \int_{-\infty}^{\infty} P(x) \log \frac{P(x)}{Q(x)}dx$$

Traditionally, we often let Q be the standard normal distribution $N(0,1)$, and P is the real mapped distribution of our hidden variable values, assume we have m input images:

$$P(x)\sim \sum_i^m N(\mu_i,\sigma_i^2) \quad Q(x) \sim N(0,1)$$

The optimization process is quite like using stochastic gradient descent: everytime we only use a batch of samples to calculate the P(x), so this is a stochastic optimization. In the mean sense, a normal distribution Q can be reached by P.

<img src="https://github.com/tensorstone/markdownfigs/blob/master/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-08-19%2019.10.52.png?raw=true" width=550 height=550 />

As a matter of fact, the KL-term here act as a regularizer that can restrict V_mean and V_variance. The final result is we can use any point from the Gaussian in this hidden space to generate a quite good image that at least looks like one of input samples of the several classes.

This N(0,1) priori is good at generating new images, but not conducive to unsupervised classification. To separate different classes into two or more clusters, a better choice of the priori distribution can be a double-peak Gaussian.

$$Q(x)\sim \frac{1}{2}N(-m,s^2)+\frac{1}{2}N(m,s^2)$$

<img src="https://github.com/tensorstone/markdownfigs/blob/master/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-08-19%2019.11.56.png?raw=true" width=550 height=550 />

### 1.1.2 The calculation of $D_{KL}(P,Q)$
#### 1.1.2.1 An scaling technique
\begin{align*}
&D_{KL}\left(N(\mu,\sigma^2) \Vert \frac{1}{2}N(-m,s^2)+\frac{1}{2}N(m,s^2)\right)\\
=&D_{KL}\left(2*\frac{1}{2}N(\mu,\sigma^2)\Vert \frac{1}{2}N(-m,s^2)+\frac{1}{2}N(m,s^2)\right)
\\
\leq & D_{KL}\left(\frac{1}{2}N(\mu,\sigma^2)\Vert \frac{1}{2}N(-m,s^2)\right)+D_{KL}\left(\frac{1}{2}N(\mu,\sigma^2)\Vert \frac{1}{2}N(m,s^2)\right)
\\
=& -\frac{1}{2}\log{2}\left( s^2 + \log{\sigma^2} -\frac{1}{2}(\mu-m)^2 -\frac{1}{2}(\mu+m)^2 -\sigma^2 \right)
\end{align*}

#### 1.1.2.2 More precise calculation
\begin{align*}
&D_{KL}\left(N(\mu,\sigma^2) \Vert \frac{1}{2}N(-m,s^2)+\frac{1}{2}N(m,s^2)\right)\\
=&\int_{-\infty}^{\infty}N(\mu,\sigma^2) \log{\frac{N(\mu,\sigma^2) }{\frac{1}{2}N(-m,s^2)+\frac{1}{2}N(m,s^2)}}dx\\
=&\int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}\log{\frac{\frac{1}{\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}}{\frac{1}{2s}[e^{-\frac{(x-m)^2}{2s^2}}+e^{-\frac{(x+m)^2}{2s^2}}]}}dx\\
=& \int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}\log{\frac{2s}{\sigma}}dx - \int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}[\frac{(x-\mu)^2}{2\sigma^2}-\frac{(x-m)^2}{2s^2}] dx - \int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}\log{[1+e^{-\frac{2mx}{s^2}}]}dx\\
=& \alpha - \beta - \gamma \\
&\quad \\
\alpha &= \log{\frac{2s}{\sigma}}\\
\beta &= -\frac{(m-\mu)^2+\sigma^2 - s^2}{2s^2}\\
\gamma &\approx -\frac{2m[-\sigma e^{-\frac{\mu^2}{2\sigma^2}}+\sqrt{\frac{\pi}{2}}\mu Erfc(\frac{\mu}{\sqrt{2}\sigma})]}{s^2}
\end{align*}
with an approximation of $Erfc(x) \approx 1-tanh(1.19x)$
$$D_{KL}\left(N(\mu,\sigma^2) \Vert \frac{1}{2}N(-m,s^2)+\frac{1}{2}N(m,s^2)\right) \approx\log{\frac{2s}{\sigma}}+\frac{(m-\mu)^2+\sigma^2 - s^2}{2s^2} + \frac{2m\{-\sigma e^{-\frac{\mu^2}{2\sigma^2}}+\sqrt{\frac{\pi}{2}}\mu [1-tanh(1.19\frac{\mu}{\sqrt{2}\sigma})]\}}{s^2}$$



# Double-peak KL loss:
def vae_loss(x, decoded):  
    xent_loss = K.sum(K.sum(objectives.binary_crossentropy(x ,decoded),axis=-1),axis=-1)
    #kl_loss_d1 = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) 
    m = K.constant(0)
    s = K.constant(1)
    # the following line: when m=0,s=1, it should be a single peak normal distribution
    #kl_loss_d1 = K.sum(K.log(2/K.exp(z_log_var/2))+(K.square(z_mean)+(K.exp(z_log_var/2)-K.constant(1))*(K.exp(z_log_var/2)+K.constant(1)))/(K.constant(2)),axis = -1)
    kl_loss_d1 = K.sum(K.log(2*s/K.exp(z_log_var/2))+(K.constant(2)*m*(-K.exp(-(K.square(z_mean))/((K.constant(2))*K.exp(z_log_var)))*K.exp(z_log_var/2) + K.sqrt(K.constant(np.pi/2))*z_mean*(K.constant(1)-K.tanh(K.constant(1.19)*z_mean/K.constant(np.sqrt(2))/K.exp(z_log_var/2)))) )/(K.square(s))+(K.square(m-z_mean)+(K.exp(z_log_var/2)-s)*(K.exp(z_log_var/2)+s))/(K.constant(2)*K.square(s)),axis = -1)
    return 1*xent_loss + 1*kl_loss_d1 
    
The figure below shows this fitting process: (a) is the result when we minimize $D_{KL}(Q,P)$, (b)(c) are the results when we minimize $D_{KL}(P,Q)$
<img src="https://github.com/tensorstone/markdownfigs/blob/master/pq1.png?raw=true" width=550 height=550 />

### 1.1.3 Defect of KL-divergence and analogies of Wasserstein loss
#### 1.1.3.1 Defect of KL-divergence
Sometimes when we train the neural net with a KL-divergence or JS-divergence, we meet the gradient disappearence problem: in the figure below, $P_1$ and $P_2$ are two dirac $\delta$ functions (we may suppose they are two gaussian distributions with almost no overlap) 

<img src="http://attachbak.dataguru.cn/attachments/portal/201702/10/184047b3so7s8iywgw6wo3.jpg" width=250 height=250 />

The KL divergence of $P_1$ & $P_2$ is:
$$D_{KL}(P_1,P_2)=
\begin{cases}
\infty & \theta\not=0\\
 0& \theta=0
\end{cases}$$
The JS divergence of $P_1$ & $P_2$ is: 
$$f(x)=
\begin{cases}
log2&{\theta \not= 0}\\
0&{\theta=0}
\end{cases}$$

Most of the time the derivatives of KL-divergence and JS-divergence are zero.

But if we use Wasserstein metric here, $$Wasserstein loss = ||\theta||_p$$ which is continuous. This is quite like what Arjovsky et al. did in [Wasserstein GAN](http://xueshu.baidu.com/s?wd=paperuri%3A%281349733e9788e9a2049fa4615a740cc3%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fabs%2F1701.07875&ie=utf-8&sc_us=587259751049901359)


#### 1.1.3.2 A naive analogy
With $$P(x)\sim \sum_i^m N(\mu_i,\sigma_i^2) \quad Q(x)\sim \frac{1}{2}N(-m,s^2)+\frac{1}{2}N(m,s^2)$$
an naive way to avoid gradient disappearance is to introduce an analogy of Wasserstein loss (AW loss):
$$AW(P,Q) = -||\mu||_p + ||\sigma - s||_p$$
which makes $\mu$ as large as possible and constrain $\sigma$ by the second term. In practice, there should be an tanh activation function in the last layer of the encoder (so that the value of $\mu$ is limited).
#### 1.1.3.3 Another analogy
The defination of $p^{th}$ [Wasserstein metric](https://en.wikipedia.org/wiki/Wasserstein_metric) is 
$$W_p(\mu,\nu):= \left(\inf_{\gamma\in\Gamma(\mu,\nu)}\int_{M\times M}d(x,y)^pd\gamma(x,y) \right)^{1/p}$$. 

Another name of Wasserstein metric is Earth Mover Distance,which is pretty vivid. The Earth Mover Distance is talking about an analogy that we may understand the Wasserstein metric in this way: If we want to move one of a sand dune to another place, then the "W distance" is defined by the smallest work an earth mover need to do.

Back to our problem, here we need to calculate the optimal route of transportation, which is an combinatorial optimization problem. To avoid this difficulty, I tried to use another analogy. I called it Pseudo Wasserstein(PW) loss:
With $$P(x)\sim \sum_i^m N(\mu_i,\sigma_i^2) \quad Q(x)\sim \frac{1}{2}N(-m,s^2)+\frac{1}{2}N(m,s^2)$$
$$PW(P,Q) = ||\mu||_p + \left(\int_{-\infty}^{\infty}||P-Q||_pdx\right)^{1/p} $$

By this definition, 
\begin{align*}
&PW\left(N(\mu,\sigma^2) \Vert \frac{1}{2}N(-m,s^2)+\frac{1}{2}N(m,s^2)\right)\\
=&||\mu||_p + \left(\int_{-\infty}^{\infty}|| N(\mu,\sigma^2)-[\frac{1}{2}N(-m,s^2)+\frac{1}{2}N(m,s^2)] ||_p dx\right)^{1/p}\\
& \quad  \\
=& \frac{1}{4\sqrt{\pi}s\sigma\sqrt{\frac{1}{s^2}+\frac{1}{\sigma^2}}}\exp{\left(-\frac{m^2}{s^2}-\frac{\mu^2}{2(s^2+\sigma^2)}\right)}\\
&\left[-2\sqrt{2}\exp{\left(\frac{m[-2\mu s^2 + m(s^2 + 2\sigma^2)]}{2s^2(s^2+\sigma^2)}\right)}- 2\sqrt{2}\exp{\left(\frac{m[2\mu s^2 + m(s^2 + 2\sigma^2)]}{2s^2(s^2+\sigma^2)}\right)}  + \sigma\sqrt{\frac{1}{s^2}+\frac{1}{\sigma^2}} \exp{\left( \frac{\mu^2}{2(s^2+\sigma^2)}\right)} + \sigma s (\frac{1}{s}+\frac{2}{\sigma}\sqrt{\frac{1}{s^2}+\frac{1}{\sigma^2}})\exp{ \frac{m^2}{s^2}+\frac{\mu^2}{2(s^2+\sigma^2)} } \right]
\end{align*}

# 1.2 Experiments
## 1.2.1 Benchmark when using traditional KL loss
### 1.2.1.1 AE
To demonstrate the KL term in the loss function, the hidden variables of AEs designed for this task is shown first.

The figures below shows the hidden variable distribution when I use AE and include Batch Normalization(BN) after each activation function in the encoder. In each figure, the X-axis is the value of the first hidden variable while Y-aixs is the second one. I chose to use 2 hidden variables in this classification task. The selection of hidden variable number will be discussed later in Part4 of this notebook.
<img src="https://github.com/tensorstone/markdownfigs/blob/master/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202017-08-20%2001.20.22.png?raw=true" width=650 height=650 />

The purple points represent stars, the green points represent galaxies and the yellow points are for QSOs. Without KL term and resample process in the hidden layer, the hidden space geometry is very flat.

[Unsupervised learning of phase transitions: from principal component analysis to variational autoencoders](https://arxiv.org/abs/1703.02435) introduced an unsupervised learning way to approach phase transitions. The author used PCA, AE and VAE to learn about the latent parameters which best describe states of the two-dimensional Ising model and the three-dimensional XY model. And by using only one hidden variable and compare it to magnetic moment values, the author draw the conclusion that the linear relation between hidden variable and magnetic moment shows that they are the same thing. In other word, the neuralnet learned the order parameter that can best separate different classes(two phases of Ising model).

# [TODO: physical correspondance]

### 1.2.1.2 VAE
As a comparision, the hidden variables of VAEs designed for this task is shown below:
<img src="https://github.com/tensorstone/markdownfigs/blob/master/%E4%B8%8B%E8%BD%BD.png?raw=true" width=650 height=650 />

The spiral structure comes from the KL term introduced in Part 1.1.1

If we run manifold learning on this hidden space, we may be able to perform classification. We use manifold learning here for different classes may stick to each other in the hidden space, so that the most useful clustering criteria should be the geometry information.

However, although the spiral hidden space structure seems prone to be separated into two clusters, this is an undertrained model. If we train the model for more epochs, the result would be:
#### Hidden variable distribution
<img src="https://github.com/tensorstone/Galaxy_Zoo/blob/master/unstable.png?raw=true" width=650 height=650 />

whose distribution is more likely to obey a standard normal distribution.
#### ISOMAP on hidden space
If we run manifold learning(I use [ISOMAP](http://scikit-learn.org/stable/modules/manifold.html)) on this hidden space, we will get the second image in the figure below:
<img src="https://github.com/tensorstone/Galaxy_Zoo/blob/master/manif_.png?raw=true" width=650 height=650 />
#### ROC curve
And we can then use groundtruth labels to draw the ROC curve and calculate the AUC:
<img src="https://github.com/tensorstone/Galaxy_Zoo/blob/master/ROC_benchmark.png?raw=true" width=350 height=350 />
#### AUC = 0.731


## 1.2.2 Comparison of different penalty functions
### 1.2.2.1 KL-a scaling technique
#### Hidden variable distribution
<img src="https://github.com/tensorstone/markdownfigs/blob/master/scaling_hidden.png?raw=true" width=650 height=650 />
#### ISOMAP on hidden space
<img src="https://github.com/tensorstone/markdownfigs/blob/master/scaling_isomap.png?raw=true" width=650 height=650 />
#### ROC curve
<img src="https://github.com/tensorstone/markdownfigs/blob/master/scaling_ROC.png?raw=true" width=350 height=350 />
#### AUC = 0.821

### 1.2.2.2 KL-more precise calculation
#### Hidden variable distribution
<img src="https://github.com/tensorstone/markdownfigs/blob/master/DKL_hidden.png?raw=true" width=650 height=650 />
#### ISOMAP on hidden space
<img src="https://github.com/tensorstone/markdownfigs/blob/master/DKL_isomap.png?raw=true" width=650 height=650 />
#### ROC curve
<img src="https://github.com/tensorstone/markdownfigs/blob/master/DKL_ROC.png?raw=true" width=350 height=350 />
#### AUC = 0.910



### 1.2.2.3 Analogy of Wasserstein Loss
#### Hidden variable distribution
<img src="https://github.com/tensorstone/Galaxy_Zoo/blob/master/AW_hidden.png?raw=true" width=650 height=650 />
#### ISOMAP on hidden space
<img src="https://github.com/tensorstone/Galaxy_Zoo/blob/master/AW_ISOMAP.png?raw=true" width=650 height=650 />
#### ROC curve
<img src="https://github.com/tensorstone/Galaxy_Zoo/blob/master/AW_ROC.png?raw=true" width=350 height=350 />
#### AUC = 0.894


### 1.2.2.4 Pseudo Wasserstein Loss
### 1.2.2.3 Analogy of Wasserstein Loss
#### Hidden variable distribution
<img src="https://github.com/tensorstone/markdownfigs/blob/master/PW_1,1_hidden.png?raw=true" width=650 height=650 />
#### ISOMAP on hidden space
<img src="https://github.com/tensorstone/markdownfigs/blob/master/PW_1,1_isomap.png?raw=true" width=650 height=650 />
#### ROC curve
<img src="https://github.com/tensorstone/markdownfigs/blob/master/PW_1,1_ROC.png?raw=true" width=350 height=350 />
#### AUC = 0.875


# Part2. Unsupervised Star-Galaxy/Background Segmentation
## 2.1 Hypercolumns: Extracting Features for Pixel-level Segmentation
Here we used [Hypercolumns (by Hariharan et al.)](https://arxiv.org/abs/1411.5752) to combine the features extracted by AEs/VAEs.

In the segmentation task, the variational part in VAEs can not help to improve the neural net's performance, for we don't aim at using any point in the hidden space to generate images that look real. To obtain better performance, I tried to use residual connections between each down-sampling layer and the correspond up-sampling layer.
<img src="https://github.com/tensorstone/markdownfigs/blob/master/hypercolumn.png?raw=true" width=700 height=700 />

### 2.1.1 VAEs + Hypercolumns
<img src="https://github.com/tensorstone/markdownfigs/blob/master/VAE30hidden.png?raw=true" width=700 height=700 />
When using VAE, there are some shape distortion and blurred edge. Those are not conducive to pixel-level segmentation.

### 2.1.2 VAEs with ConvReplacePooling + Hypercolumns

<img src="https://github.com/tensorstone/markdownfigs/blob/master/allconvpoolingearlystop.png?raw=true" width=700 height=700 />
If we use convolution layers with $2\times 2$ stides to replace Maxpooling layers, the result can be improved a little for more structral informations are saved.
### 2.1.3 AEs + Hypercolumns
<img src="https://github.com/tensorstone/markdownfigs/blob/master/AE_reproduction.png?raw=true" width=700 height=700 />
If we use AE instead of VAE, most stars and galaxies can be found. But some of the very faint objects are missed.


### 2.1.4 AEs + Residual connections + Hypercolumns

<img src="https://github.com/tensorstone/markdownfigs/blob/master/skipconnectionearlystop.png?raw=true" width=700 height=700 />
If we use AEs with residual connections, then more structure informations can be kept in the output layer. In the first image of the figure above, the spiral structure is also reproduced.

However, more noises are also included in output images as a cost. In practice, we may change the layers we choose to use and turn their weights as a trade-off.


## 2.2 Summary and challenge
### 2.2.1 A trade-off  between noises and faint stars 
### 2.2.2 Standard or benchmark: supervised segmentation?

# Part3. Unsupervised Pixel-wise Star-Galaxy Segmentation and Classification

Combine Part1 and Part2, we can assign each pixel with a label wether it belongs to an object. If the answer is yes, then if the pixel and its neighbors are recognized as a star/galaxy, this pixel will be assigned as a star/galaxy. The graph below shows the model architecture:

<img src="https://github.com/tensorstone/Galaxy_Zoo/blob/master/Structurefig.001.jpeg?raw=true" width=850 height=850 />


# Part4. On the Number of Hidden Variables
In the experiment I took, I just found the most important thing is not about the hidden variable number nor about the sample size.

In deeplearning, it's always useful to have more data in the training process. If we have more training data, we will be able to locate the "support vecters" more precisely. And then get better classification performance.

However, in unsupervised learning, things are quite different: when we use lots of datas in the training process, different classes may mapped to a certain neighbourhood(a certain small area). The hidden variables are likely to "stick" to each other. What we want to do here is unsupervised learning, or clustering. 

When we use manifold learning or any other geometry based clustering method, the only useful information we may use to separate different classes is the gap between them. 
# Part5. TODOs:
## 5.1 Hyperparameters: m,s in KL/Wasserstein divergence
## 5.2 Physical correspondance
## 5.3 Benchmark of segmentation task
## 5.4 Better manifold learning algorithms?
