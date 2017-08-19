# Galaxy_Zoo

Aim
-----
In this project, I want to find a unsupervised way to seperate galaxies and stars.
TODOs:
### 1.Use Galaxy Zoo data
VAE + HypCol vs. CNNs + surrogate class + HypCol

https://github.com/EdwardJKim/unsupervised-dl4astro/blob/master/notebooks/kmeans.ipynb
### 2.On SDSS datasets




gala.ipynb
-----
on galaxy zoo dataset

SDSS.ipynb
----
on SDSS dataset

cutfigure.py
----
to cut figures of galaxy zoo dataset

SDSS_clustering.ipynb
----
use manifold learning method to imporve the accuracy of classification result.
use both 2D and 3D hidden variables as inputs

TC(1).ipynb
----
use several different VAE structures to help clustering/hidden layer classification process


initial_normalization.ipynb
----
Kmeans clustering with the original normalization method(-mean,/max)


RepW4.ipynb
----
Report for week 4

some of the code is credit to Edward Kim:

https://github.com/EdwardJKim/unsupervised-dl4astro/


Find+physics_failed.ipynb
---
Compare size/luminosity of objects and the hidden variables

seems only one hidden variable is useful

can not use such hidden variables to predict the redshift 

15epoch_3hidden_newone0808.h5
---
Model saved in Classification+result+is+better+when+I+use+5+channels.ipynb

Classification+result+is+better+when+I+use+5+channels.ipynb
----
A good Classification result with ROC curve

The overall acc reached 88.8%

Correct+normalization+%26+great+segmentation.ipynb
----
together with the saved parameters: Correct_normalization_great_segmentation0807.h5.zip

Only use the 4th channel (i channel) instead use all of the 5 channels in segmentation. Got much better result than before


RepW6_high_performance_result.ipynb,(1) is the most recent one
---
Reached the highest accuracy of 88.8%

Combined the results by Aug.8
![image](https://github.com/tensorstone/Galaxy_Zoo/blob/master/FIG.001.jpeg)

Try_more_hidden_variables.ipynb
-----
Better segmentation result when I use AE instead of VAE.(It should be like this)

and some result are saved in OneNote


A Pseudo Wasserstein loss.ipynb
----
The result here is pretty like when I used L2 regularizer

Accurate_KL_and_Pseudo_Wasserstein_loss.ipynb
-----
Compared several reasonable loss functions.

I calculate those loss function using Mathematica. The .nb files are also uploaded
