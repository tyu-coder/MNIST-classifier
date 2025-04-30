# 项目名称：构建神经网络实现手写数字MNIST图像分类

# 项目简介：
本项目在首先实现基本神经网络分类MNIST图像功能的基础上，从多个方面探讨优化模型的方法，最终完成MNIST数字图像的分类，使得在测试集上的准确率可以达到90%左右

# 目录结构
| ———best_models //储存最终最优模型参数文件的文件夹
|   | ———best_model.pickle//最优模型参数文件
| ———best_models_conv //储存卷积网络训练的最优模型参数文件的文件夹
|   | ———best_model.pickle//卷积网络最优模型参数文件
| ———best_models_dropout //储存使用dropout的MLP网络训练的最优模型参数文件的文件夹
|   | ———best_model.pickle//使用dropout的MLP网络最优模型参数文件
| ———best_models_dropout //储存使用L2正则化的MLP网络训练的最优模型参数文件的文件夹
|   | ———best_model.pickle//使用L2正则化的MLP网络最优模型参数文件
| ———dataset//储存MNIST数据集的文件夹
| ———draw_tools//储存绘制结果图像代码的文件夹
| ———mynn//储存神经网络模型基本代码的文件夹
|   | ———_pycache_
|   | ———_init_.py
|   | ———data_aug.py//数据增强代码
|   | ———lr_scheduler.py//调节学习率的代码
|   | ———metric.py//计算准确率的代码
|   | ———models.py//实现MLP和CNN类的代码
|   | ———op.py//实现网络基本层比如linear层等的代码
|   | ———optimizer.py//优化器的代码
|   | ———runner.py//控制运行和输出的代码
| ———saved_models_[128, 64, 32]//储存nHidden=[128, 64, 32]时的最优模型参数文件的文件夹
|   | ———best_model.pickle//nHidden=[128, 64, 32]时的最优模型参数文件
| ———saved_models_[128, 64]//储存nHidden=[128, 64]时的最优模型参数文件的文件夹
|   | ———best_model.pickle//nHidden=[128, 64]时的最优模型参数文件
| ———saved_models_[256,128]//储存nHidden=[256,128]时的最优模型参数文件的文件夹
|   | ———best_model.pickle//nHidden=[256,128]时的最优模型参数文件
| ———saved_models_[512]//储存nHidden=[512]时的最优模型参数文件的文件夹
|   | ———best_model.pickle//nHidden=[512]时的最优模型参数文件
| ———dataset_explore.ipynb
| ———idx.pickle
| ———README.md
| ———test_model.py//在测试集上测试的代码
| ———test_train_conv.py//实现卷积网络训练的代码
| ———test_train_nHidden.py//实现不同结构的MLP模型训练的代码
| ———test_train_regularization.py//带L2正则化，dropout，early-training的MLP模型训练的代码
| ———test_train.py//实现最基本的模型训练的代码
| ———training_log_conv.txt //卷积网络训练结果记录
| ———training_log_ dropout.txt //带dropout，early-training的MLP模型训练结果记录
| ———training_log_ L2.txt //带L2正则化，early-training的MLP模型训练结果记录
| ———training_log_ neither.txt //只带 early-training的MLP模型训练结果记录
| ———training_log.txt //最基本的模型训练结果记录
| ———weight_visualization.py

# 使用说明
运行test_train.py可以对最基本的网络进行训练
运行test_train_conv.py可以对卷积网络的参数进行设置并进行卷积网络的训练
运行test_train_nHidden.py可以对不同nHidden的MLP模型训练，也可以对nHidden进行调节
运行test_train_regularization.py 单独设置用来看使用正则化与否的情况，但实际使用的时候要修改models中的参数dropout_p和weight_decay
运行test_model.py来看模型在测试集上表现，不过要注意对best_model的选择

# 结果说明
我们在给定代码的基础上基本实现了神经网络模型对MNIST数据集进行分类的问题，并且在基础模型之上对于是否正则化、使用怎样的网络结构、是否使用卷积网络等方向上进行了改进和研究，最好的测试结果在测试集上的准确率达到了0.9091
但是受到算力有限的影响，本实验理论上应该还需要做以下改进：对更多的参数，比如说学习率，正则化系数进行调整；每个模型进行更多个epoch的实验训练；增大卷积核大小进行实验
