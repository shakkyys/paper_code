import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.optimizers import SGD
import matplotlib.pyplot as plt

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os


def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None):
    """
    标准神经网络训练过程。
    """
    model = Sequential()

    print(data.train_data.shape)
    
    model.add(Conv2D(params[0], (3, 3), input_shape=data.train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(10))
    
    if init is not None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])  # 使用'accuracy'指标

    # 进行模型训练
    history = model.fit(data.train_data, data.train_labels,
                        batch_size=batch_size,
                        validation_data=(data.validation_data, data.validation_labels),
                        epochs=num_epochs,
                        shuffle=True)
    
    if file_name is not None:
        model.save(file_name)

    return model, history


def train_distillation(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1):
    """
    使用防御蒸馏进行模型训练。

    Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
    Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
    IEEE S&P, 2016.
    """
    if not os.path.exists(file_name+"_init"):
        # 仅训练一个周期以获得一个良好的起点
        train(data, file_name+"_init", params, 1, batch_size)
    
    # 现在在给定的温度下训练teacher模型
    teacher, history_teacher = train(data, file_name + "_teacher", params, num_epochs, batch_size, train_temp,
                                     init=file_name + "_init")

    # 在温度t下评估标签
    predicted = teacher.predict(data.train_data)
    with tf.compat.v1.Session() as sess:
        y = sess.run(tf.nn.softmax(predicted/train_temp))
        print(y)
        data.train_labels = y

    # 在温度t下训练student模型
    student, history_student = train(data, file_name, params, num_epochs, batch_size, train_temp,
                                     init=file_name+"_init")

    # 最后在温度1下进行预测
    predicted = student.predict(data.train_data)

    print(predicted)
    
    return student, history_student, history_teacher
    
    
def plot_training_curves(history, file_name):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(file_name)


if not os.path.isdir('models'):
    os.makedirs('models')

# ...

cifar_model, cifar_history = train(CIFAR(), "models/cifar_model", [64, 64, 128, 128, 256, 256], num_epochs=50)
mnist_model, mnist_history = train(MNIST(), "models/mnist_model", [32, 32, 64, 64, 200, 200], num_epochs=50)

mnist_student, mnist_distilled_history, _ = train_distillation(
    MNIST(), "models/mnist-distilled-100_model", [32, 32, 64, 64, 200, 200], num_epochs=50, train_temp=100)

cifar_student, cifar_distilled_history, _ = train_distillation(
    CIFAR(), "models/cifar-distilled-100_model", [64, 64, 128, 128, 256, 256], num_epochs=50, train_temp=100)

# 将训练曲线保存为图片
if not os.path.exists('models/train_images'):
    os.makedirs('models/train_images')

plot_training_curves(cifar_history, "models/train_images/cifar_training_curve.png")
plot_training_curves(mnist_history, "models/train_images/mnist_training_curve.png")
plot_training_curves(mnist_distilled_history, "models/train_images/mnist_distilled_training_curve.png")
plot_training_curves(cifar_distilled_history, "models/train_images/cifar_distilled_training_curve.png")
