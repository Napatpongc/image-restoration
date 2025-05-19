"""
Denoise an image with the FFDNet denoising method

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import FFDNet
from utils import batch_psnr, normalize, init_logger_ipol, \
                  variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_ffdnet(**args):
    # init logger
    logger = init_logger_ipol()

    # check if input is RGB or grayscale
    try:
        rgb_den = is_rgb(args['input'])
    except Exception as e:
        raise RuntimeError('Cannot open image file') from e

    # load image as numpy CxHxW
    if rgb_den:
        in_ch   = 3
        model_fn= 'models/net_rgb.pth'
        imorig  = cv2.imread(args['input'])
        imorig  = cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB).transpose(2,0,1)
    else:
        in_ch   = 1
        model_fn= 'models/net_gray.pth'
        imorig  = cv2.imread(args['input'], cv2.IMREAD_GRAYSCALE)
        imorig  = np.expand_dims(imorig, 0)
    imorig = np.expand_dims(imorig, 0)

    # pad odd-size dimensions
    sh = imorig.shape
    expanded_h = sh[2] % 2 == 1
    expanded_w = sh[3] % 2 == 1
    if expanded_h:
        imorig = np.concatenate((imorig, imorig[:, :, -1:, :]), axis=2)
    if expanded_w:
        imorig = np.concatenate((imorig, imorig[:, :, :, -1:]), axis=3)

    # normalize and to torch.Tensor
    imorig = normalize(imorig)
    imorig = torch.Tensor(imorig)

    # load model weights
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)
    print('Loading model:', model_path)
    net = FFDNet(num_input_channels=in_ch)
    if args['cuda']:
        state_dict = torch.load(model_path)
        model      = nn.DataParallel(net, device_ids=[0]).cuda()
    else:
        state_dict = torch.load(model_path, map_location='cpu')
        state_dict = remove_dataparallel_wrapper(state_dict)
        model      = net
    model.load_state_dict(state_dict)
    model.eval()

    # set dtype for tensors
    dtype = torch.cuda.FloatTensor if args['cuda'] else torch.FloatTensor

    # create noisy input or clone original
    if args['add_noise']:
        noise   = torch.FloatTensor(imorig.size()).normal_(mean=0, std=args['noise_sigma'])
        imnoisy = imorig + noise
    else:
        imnoisy = imorig.clone()

    # inference
    with torch.no_grad():
        imorig, imnoisy = Variable(imorig.type(dtype)), Variable(imnoisy.type(dtype))
        nsigma          = Variable(torch.FloatTensor([args['noise_sigma']]).type(dtype))

    start_t = time.time()
    im_noise_estim = model(imnoisy, nsigma)
    outim          = torch.clamp(imnoisy - im_noise_estim, 0., 1.)
    stop_t  = time.time()

    # remove padding
    if expanded_h:
        outim = outim[:, :, :-1, :]
    if expanded_w:
        outim = outim[:, :, :, :-1]

    # save output image
    if not args['dont_save_results']:
        out_path = args['output']
        out_img  = variable_to_cv2_image(outim)
        cv2.imwrite(out_path, out_img)
        # print path for wrapper to capture
        print(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFDNet_Test")
    parser.add_argument('--add_noise',    type=str,   default="True")
    parser.add_argument("--input",        type=str,   required=True, help='path to input image')
    parser.add_argument("--noise_sigma",  type=float, default=25,   help='noise level (0-255)')
    parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
    parser.add_argument("--no_gpu",       action='store_true', help="run model on CPU")
    parser.add_argument("--output",       type=str,   required=True, help='path to save denoised image')

    argspar = parser.parse_args()
    # normalize sigma to [0,1]
    argspar.noise_sigma /= 255.0
    argspar.add_noise    = (argspar.add_noise.lower() == 'true')
    argspar.cuda         = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FFDNet model ###")
    for k, v in vars(argspar).items():
        print(f"\t{k}: {v}")
    print()

    test_ffdnet(**vars(argspar))
