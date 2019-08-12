# Face Generation Using Variational Autoencoders
This repo contains training code for two different VAEs implemented with Pytorch. <br />
I used the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) Dataset for training, with 182637 training images and 19962 testing images. <br />
Trained model can be found in [/checkpoints](/checkpoints).

![poster](arts/vae.png)

## Model structures:

#### [β-VAE [1]](https://openreview.net/pdf?id=Sy2fzU9gl):
![beta-vae](arts/beta-vae.png)

#### [DFC-VAE [2]](https://arxiv.org/abs/1610.00291):
![dfc-vae](arts/dfc-vae.png)


## Results after 300 epochs:

##### Original Faces (Top) vs. Reconstructed (Bottom) :
![r1](arts/297-dh.png) ![r2](arts/301-dh.png)
![r3](arts/302-dh.png) ![r4](arts/303-dh.png)

##### Linear Interpolation from z1 (leftmost) to z2 (rightmost):
![l1](arts/interpolate-dh.png)

##### Vector Arithmetic from original (leftmost) to wearing sunglasses (rightmost):
![l1](arts/arithmetic-dfc2-dh.png)

##### Generated Images with randomly sampled latent z ~ N(0, 1):
![l1](arts/dfc-300-dh.png)

(Notes: output images above are results after image dehazing using [this script [3]](https://github.com/cssartori/image-dehazing.git))

## References
[1] β-VAE: https://openreview.net/pdf?id=Sy2fzU9gl <br />
[2] DFC-VAE: https://arxiv.org/abs/1610.00291 <br />
[3] Dehaze: https://github.com/cssartori/image-dehazing.git
