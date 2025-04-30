# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import sys
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

def safe_augment(imgs):
      augmented_imgs=imgs.copy()
      if np.random.rand()<0.7:
            augmented_imgs=random_shift(augmented_imgs)
      if np.random.rand()<0.7:
            augmented_imgs=random_rotation(augmented_imgs)    
      if np.random.rand()<0.7:
            augmented_imgs=add_noise(augmented_imgs)
      augmented_imgs=np.clip(augmented_imgs,0.,1.)  
      return augmented_imgs        
log_file=open('training_log_2.txt','w',buffering=1,encoding='utf-8')
sys.stdout=log_file
sys.stderr=log_file
# fixed seed for experiment
np.random.seed(109)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / 255.
valid_imgs = valid_imgs / 255.

nHidden=[256,128]

layer_sizes=[train_imgs.shape[-1]]+nHidden+[10]

linear_model = nn.models.Model_MLP(layer_sizes, 'ReLU', [1e-3 for _ in range(len(nHidden)+1)])
my_optimizer = nn.optimizer.MomentGD(init_lr=0.01, model=linear_model,mu=0.9)
my_scheduler = nn.lr_scheduler.MultiStepLR(optimizer=my_optimizer, milestones=[800, 2400, 4000], gamma=0.9)
my_loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(model=linear_model,optimizer=my_optimizer, metric=nn.metric.accuracy, loss_fn=my_loss_fn, scheduler=my_scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models_L2')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()

log_file.close()
sys.stdout=sys.__stdout__
sys.stderr=sys.__stderr__