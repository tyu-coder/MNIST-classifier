import numpy as np
import scipy.ndimage

def random_shift(imgs,max_shift=2):
    shifted_imgs=np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        dx=np.random.randint(-max_shift,max_shift+1)
        dy=np.random.randint(-max_shift,max_shift+1)
        shifted_imgs[i]=np.roll(imgs[i],shift=(dx,dy),axis=(0,1))
    return shifted_imgs
def random_rotation(imgs,max_angle=15):
    rotated_imgs=np.zeros_like(imgs)
    for i in range(imgs.shape[0]):
        angle=np.random.uniform(-max_angle,max_angle)
        rotated_imgs[i]=scipy.ndimage.rotate(imgs[i],angle,reshape=False,mode='nearest')
    return rotated_imgs
def add_noise(imgs,noise_level=0.1):
    noise=np.random.normal(0,noise_level,imgs.shape)
    noisy_imgs=imgs+noise
    return np.clip(noisy_imgs,0.,1.)
