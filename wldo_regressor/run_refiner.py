import sys
sys.path.append("../")

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from global_utils.refiner_config import get_config
from wldo_regressor.refiner import Refiner
from dataset_dogs.demo_dataset import DemoDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='../data/pretrained/3501_00034_betas_v4.pth', help='Path to network checkpoint')
parser.add_argument('--src_dir', default="../example3_imgs", type=str, help='The directory of input images')
parser.add_argument('--result_dir', default='../demo_out', help='Where to export the output data')
parser.add_argument('--shape_family_id', default=-1, type=int, help='Shape family to use')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--gpu_ids', default="0", type=str, help='GPUs to use. Format as string, e.g. "0,1,2')

colors = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0],
                            [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])


def write_to_txt(joints, filename):
    output_txt = '../text_files/' + filename
    with open(output_txt, 'a') as f:
        for i in range(0, len(joints)):
            xyz = joints[i]
            f.write(str(xyz[0]) + ",\t" + str(xyz[1]) + ",\t" + str(xyz[2]))
            if i < (len(joints) - 1):
                f.write(",\t")
        f.write('\n')

def plot_joints(joints, count):
    plt.cla()
    ax.scatter(joints[:, 0], joints[:, 2], -joints[:, 1], c=colors)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(0, 0)
    ax.set_xlim(-.75, 0.0)
    ax.set_ylim(-1, 0.5)
    ax.set_zlim(-.5, 0.5)
    plt.savefig('../outputs/' + str(count) + '.png')

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    # print (os.environ['CUDA_VISIBLE_DEVICES'])

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    assert torch.cuda.device_count() <= 1, "Currently up to 1 GPU is supported"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_config()
    # frames_dir = '/home/masselmeier/Desktop/SP22/CS8803/WLDO/example4_imgs'
    frames = os.listdir(args.src_dir)
    num_frames = len(frames)
    print('num frames: ', num_frames)
    dataset = DemoDataset(args.src_dir)

    refiner = Refiner(dataset, config, num_frames, args, device)

    train_data_loader = DataLoader(dataset, batch_size=num_frames, shuffle=False)  # , num_workers=num_workers)

    tqdm_iterator = tqdm(train_data_loader, desc='train', total=len(train_data_loader))

    # predict:
    count = 0
    for step, batch in enumerate(tqdm_iterator):
        loss = refiner.train(batch)


    '''
    pred_data_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # , num_workers=num_workers)

    tqdm_iterator = tqdm(pred_data_loader, desc='Eval', total=len(pred_data_loader))

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    output_txt_filename = 'mod_joints.txt'
    open('../text_files/' + output_txt_filename, "w").close()

    # predict:
    count = 0
    for step, batch in enumerate(tqdm_iterator):
        orig_Jsmal, Jsmal = refiner.predict(batch)
        #print('original Jsmal: ', orig_Jsmal)
        #print('refined Jsmal: ', Jsmal)
        new_joints = torch.squeeze(orig_Jsmal).detach().numpy()
        # print('joints size: ', np.shape(new_joints))

        # plot_joints(new_joints, count)
        write_to_txt(new_joints, output_txt_filename)
        count += 1
    '''