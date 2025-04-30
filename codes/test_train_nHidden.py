# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import sys
import os
log_file=open('training_log_1.txt','w',buffering=1,encoding='utf-8')
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

nHidden_choices=[
        [256,128],
        [128,64],
        [512],
        [128,64,32],
]
save_dir='saved_models'
os.makedirs(save_dir,exist_ok=True)
best_accuracy=0
best_model=None
best_nHidden=None
for nHidden in nHidden_choices:
        save_dir=f"./saved_models_{str(nHidden)}"
        os.makedirs(save_dir,exist_ok=True)
        print(f"Training model with hidden layers:{nHidden}")
        layer_sizes=[train_imgs.shape[-1]]+nHidden+[10]
        linear_model = nn.models.Model_MLP(layer_sizes, 'ReLU', [1e-3 for _ in range(len(nHidden)+1)])
        my_optimizer = nn.optimizer.SGD(init_lr=0.01, model=linear_model)
        my_scheduler = nn.lr_scheduler.MultiStepLR(optimizer=my_optimizer, milestones=[800, 2400, 4000], gamma=0.9)
        my_loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

        runner = nn.runner.RunnerM(model=linear_model,optimizer=my_optimizer, metric=nn.metric.accuracy, loss_fn=my_loss_fn, scheduler=my_scheduler)

        runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=200, save_dir=save_dir)
        final_accuracy=runner.dev_scores[-1]
        print(f"final accuracy with hidden layers:{nHidden} and accuracy:{final_accuracy}")
        '''best_model_file=os.path.join(save_dir,f"best_model_{str(nHidden)}.pkl")
        if final_accuracy>best_accuracy:
                best_accuracy=final_accuracy
                best_model=linear_model
                best_nHidden=nHidden
                print(f"New best model found with hidden layers:{nHidden} and accuracy:{final_accuracy}")
                with open(best_model_file,'wb') as f:
                          pickle.dump(linear_model,f)
                print(f"Saved best model with hidden layers:{nHidden} to {best_model_file}")'''
        _, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        _.set_tight_layout(1)
        plot(runner, axes)
        plt.show()
'''if best_model is not None:
        print(f"saving the overall best model with hidden layers:{best_nHidden} and accuracy:{best_accuracy}")
        final_best_model_path=os.path.join(save_dir,'best_model.pkl')
        with open(final_best_model_path,'wb') as f:
                pickle.dump(best_model,f)
        print(f"Saved the overall best model as {final_best_model_path}")'''
log_file.close()
sys.stdout=sys.__stdout__
sys.stderr=sys.__stderr__