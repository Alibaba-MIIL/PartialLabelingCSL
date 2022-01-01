import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import os


parser = argparse.ArgumentParser(description='Class Selective Loss for Partial Multi Label Classification.')

parser.add_argument('--model_path', type=str, default='./models/mtresnet_opim_86.72.pth')
parser.add_argument('--pic_path', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--dataset_type', type=str, default='OpenImagess')
parser.add_argument('--class_description_path', type=str, default='./data/oidv6-class-descriptions.csv')
parser.add_argument('--th', type=float, default=0.97)
parser.add_argument('--top_k', type=float, default=30)


def inference(im, model, class_list, args):
    if isinstance(im, str):
        im = Image.open(im)
    im_resize = im.resize((args.input_size, args.input_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()
    idx_sort = np.argsort(-np_output)
    # Top-k
    detected_classes = np.array(class_list)[idx_sort][: args.top_k]
    scores = np_output[idx_sort][: args.top_k]
    # Threshold
    idx_th = scores > args.th
    return detected_classes[idx_th], scores[idx_th], im, tensor_batch


def display_image(im, tags, filename):

    path_dest = "./results"
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)

    plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    # plt.rcParams["axes.titlesize"] = 10
    plt.title("Predicted classes: {}".format(tags))
    plt.savefig(os.path.join(path_dest, filename))


def main():
    print('Inference demo with CSL model')

    # Parsing args
    args = parse_args(parser)

    # Setup model
    print('Creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    print(f"Number of Classes: {args.num_classes}")
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    class_list = np.array(list(state['idx_to_class'].values()))
    print('Done\n')

    # Convert class MID format to class description
    df_description = pd.read_csv(args.class_description_path)
    dict_desc = dict(zip(df_description.values[:, 0], df_description.values[:, 1]))
    class_list = [dict_desc[x] for x in class_list]

    # Inference
    print('Inference...')
    tags, scores, im, tensor_batch = inference(args.pic_path, model, class_list, args)

    # displaying image
    print('Display results...')
    display_image(im, tags, os.path.split(args.pic_path)[1])

    # example loss calculation
    output = model(tensor_batch)
    loss_func1 = AsymmetricLoss()
    loss_func2 = AsymmetricLossOptimized()
    target = output.clone()
    target[output < 0] = 0  # mockup target
    target[output >= 0] = 1
    loss1 = loss_func1(output, target)
    loss2 = loss_func2(output, target)
    assert abs((loss1.item() - loss2.item())) < 1e-6

    plt.show()
    print('done\n')


if __name__ == '__main__':
    main()
