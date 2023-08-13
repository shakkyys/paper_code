import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import time
import os
import random

from setup_cifar import CIFAR, CIFARModel
from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

from PIL import Image


def show(img):
    """
    Show CIFAR-10 images in the console.
    """
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + 0.5) * 3
    if len(img) != 3072:
        return
    print("START")
    for i in range(32):
        print("".join([remap[int(round(x))] for x in img[i * 32:i * 32 + 32]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets instead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1, 1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start + i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start + i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

def compute_distortion(orig_imgs, adv_imgs):
    # 计算扭曲度函数，这里使用L0范数
    return np.sum(np.abs(orig_imgs - adv_imgs) > 0.5, axis=(1, 2, 3))

def generate_average_case_data(data, samples, start):
    # Average Case: Randomly select targets from incorrect classes
    inputs, targets = generate_data(data, samples=samples, targeted=True, start=start, inception=False)
    return inputs, targets

def generate_best_case_data(data, samples, start):
    # Best Case: Attack all incorrect classes and report easiest target class
    inputs, targets = generate_data(data, samples=samples, targeted=True, start=start, inception=True)
    return inputs, targets

def generate_worst_case_data(data, samples, start):
    # Worst Case: Attack all incorrect classes and report hardest target class
    inputs, targets = generate_data(data, samples=samples, targeted=False, start=start, inception=True)
    return inputs, targets


if __name__ == "__main__":
    with tf.Session() as sess:
        data, model = CIFAR(), CIFARModel("models/cifar_model", sess)

        # L0 attack
        l0_attack = CarliniL0(sess, model, max_iterations=500, initial_const=10, largest_const=15)

        attack_cases = ["Average Case", "Best Case", "Worst Case"]
        results = np.zeros((len(attack_cases), 2))  # 2 for mean and prob

        for case_index, case_name in enumerate(attack_cases):
            total_distortion = 0.0
            success_count = 0

            for j in range(len(data.test_data)):
                # 生成不同情况的数据
                if case_name == "Average Case":
                    inputs, targets = generate_average_case_data(data, samples=1, start=j)
                elif case_name == "Best Case":
                    inputs, targets = generate_best_case_data(data, samples=1, start=j)
                else:
                    inputs, targets = generate_worst_case_data(data, samples=1, start=j)

                adv = l0_attack.attack(inputs, targets)

                # Convert symbolic tensor to NumPy array using sess.run()
                adv_prediction = sess.run(model.predict(adv))
                if np.argmax(adv_prediction) != np.argmax(targets):
                    success_count += 1

                # 计算扭曲度
                total_distortion += compute_distortion(inputs, adv)

            mean_distortion = total_distortion.mean()
            success_probability = success_count / len(data.test_data)

            results[case_index, 0] = mean_distortion
            results[case_index, 1] = success_probability

            print(f"L0 Attack - {case_name}:")
            print("Mean Distortion:", mean_distortion)
            print("Success Probability:", success_probability)

        # Save results in tabular form
        np.savetxt("my_result/results_table.csv", results, delimiter=",")