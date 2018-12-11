# Face Generation Using Variational Autoencoders
This repo containing training code for two different VAEs implemented with Pytorch.

I used the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) Dataset for training, with 182637 training images and 19962 testing images.

## Model structures:

#### [β-VAE](https://openreview.net/pdf?id=Sy2fzU9gl)
![beta-vae](arts/beta-vae.png)

#### [DFC-VAE](https://arxiv.org/abs/1610.00291):
![dfc-vae](arts/dfc-vae.png)

trained model can be found in [/samples](/samples).


# References
[1] β-VAE: https://openreview.net/pdf?id=Sy2fzU9gl

[2] DFC-VAE: https://arxiv.org/abs/1610.00291
