# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import sys
log_file=open('training_log_conv.txt','w',buffering=1,encoding='utf-8')
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


train_imgs=train_imgs.reshape(-1,1,28,28)
valid_imgs=valid_imgs.reshape(-1,1,28,28)

layer_sizes=[1,8,16]
kernel_sizes=[3,3]
strides=[1,1]
paddings=[1,1]


cnn_model = nn.models.Model_CNN(size_list=layer_sizes, kernel_size_list=kernel_sizes,stride_list=strides,padding_list=paddings,act_func='ReLU')
my_optimizer = nn.optimizer.MomentGD(init_lr=0.01, model=cnn_model,mu=0.9)
my_scheduler = nn.lr_scheduler.MultiStepLR(optimizer=my_optimizer, milestones=[800, 2400, 4000], gamma=0.9)
my_loss_fn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(model=cnn_model,optimizer=my_optimizer, metric=nn.metric.accuracy, loss_fn=my_loss_fn, scheduler=my_scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=2, log_iters=100, save_dir=r'./best_models_conv')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()

log_file.close()
sys.stdout=sys.__stdout__
sys.stderr=sys.__stderr__