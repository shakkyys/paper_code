import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from model_train import *
import torch.utils.data as Data
import matplotlib.pyplot as plt
import os

#from torch.autograd.gradcheck import zero_gradients
batch_size = 10
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            if x.grad is not None:
                x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image

adver_example_by_FOOL = torch.zeros((batch_size,1,28,28)).to(device)
adver_target = torch.zeros(batch_size).to(device)
clean_example = torch.zeros((batch_size,1,28,28)).to(device)
clean_target = torch.zeros(batch_size).to(device)

for i, data in enumerate(testloader,0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    
    if i >= adver_nums / batch_size:
        break
    if i == 0:
        clean_example = inputs
    else:
        clean_example = torch.cat((clean_example, inputs), dim=0)

    cur_adver_example_by_FOOL = torch.zeros_like(inputs).to(device)

    for j in range(batch_size):
        r_rot, loop_i, label, k_i, pert_image = deepfool(inputs[j], model)
        cur_adver_example_by_FOOL[j] = pert_image

    # 使用对抗样本攻击模型
    pred = model(cur_adver_example_by_FOOL).max(1)[1]
    # print (simple_model(cur_adver_example_by_FOOL).max(1)[1])
    if i == 0:
        adver_example_by_FOOL = cur_adver_example_by_FOOL
        clean_target = labels
        adver_target = pred
    else:
        adver_example_by_FOOL = torch.cat((adver_example_by_FOOL, cur_adver_example_by_FOOL), dim=0)
        clean_target = torch.cat((clean_target, labels), dim=0)
        adver_target = torch.cat((adver_target, pred), dim=0)

print(adver_example_by_FOOL.shape)
print(adver_target.shape)
print(clean_example.shape)
print(clean_target.shape)


def adver_attack(model, adver_example, target, name):
    adver_dataset = Data.TensorDataset(adver_example, target)
    loader = Data.DataLoader(
        dataset=adver_dataset,
        batch_size=batch_size
    )
    correct_num = torch.tensor(0).to(device)
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        pred = model.forward(inputs).max(1)[1]
        num = torch.sum(pred == labels)
        correct_num = correct_num + num
    
    print('\n{} accuracy on test set is {}%'.format(name, 100 * correct_num/adver_nums))


adver_attack(model, adver_example_by_FOOL, clean_target, 'model_by_deepfool')




def plot_clean_and_adver(adver_example, adver_target, clean_example, clean_target):
    n_cols = 5
    n_rows = 5
    cnt = 1
    cnt1 = 1
    plt.figure(figsize=(n_cols * 4, n_rows * 2))

    if not os.path.exists("deepfool_image"):
        os.makedirs("deepfool_image")

    for i in range(n_cols):
        for j in range(n_rows):
            plt.subplot(n_cols, n_rows * 2, cnt1)
            plt.xticks([])
            plt.yticks([])
            plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
            clean_img = clean_example[cnt].permute(1, 2, 0).to('cpu').detach().numpy()
            adver_img = adver_example[cnt].permute(1, 2, 0).to('cpu').detach().numpy()

            # 将图像数据限制在0到1的范围内
            clean_img = np.clip(clean_img, 0, 1)
            adver_img = np.clip(adver_img, 0, 1)

            plt.imshow(clean_img, cmap='gray')
            plt.subplot(n_cols, n_rows * 2, cnt1 + 1)
            plt.xticks([])
            plt.yticks([])
            # plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
            plt.imshow(adver_img, cmap='gray')

            # 将展示的图像保存到名为“deepfool_image”的文件夹中
            clean_image_path = f"deepfool_image/clean_example_{cnt}.png"
            adver_image_path = f"deepfool_image/adver_example_{cnt}.png"
            plt.imsave(clean_image_path, clean_img, cmap='gray')
            plt.imsave(adver_image_path, adver_img, cmap='gray')

            cnt = cnt + 1
            cnt1 = cnt1 + 2

    plt.show()


plot_clean_and_adver(adver_example_by_FOOL, adver_target, clean_example, clean_target)



