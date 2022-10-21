!python train.py --batch_size 1 --dataroot ./datasets/cancer --name cancer_cyclegan --model cycle_gan --display_id -1


-------done-------
learning rate of 0.0002
normal initialization
learning rate linear decay to zero after the first 100 epochs
we use the adam optimizer with momentum 𝛽1 = 0.5 
least squares loss Hlsgan 
instance normalization


----------------------
import matplotlib.pyplot as plt

img = plt.imread('./results/horse2zebra_pretrained/test_latest/images/n02381460_1010_fake.png')
plt.imshow(img)

import matplotlib.pyplot as plt

img = plt.imread('./results/horse2zebra_pretrained/test_latest/images/n02381460_1010_real.png')
plt.imshow(img)