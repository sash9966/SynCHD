"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


import importlib.util

# BUG: Unsure why import fails, this fixed it, from chatgpt
path_to_util = "/oak/stanford/groups/amarsden/sastocke/2Dslicesfor3D/util/util.py"

spec = importlib.util.spec_from_file_location("util", path_to_util)
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)

# Now you can use the util module and call its functions

# import sys
# print(f'sys path: {sys.path}')
# sys.path.insert(0,'/home/sastocke/2Dslicesfor3D/util/util.py')
# print(f'sys path: {sys.path}')
# print('done')
# import traceback


import torch
from models.networks.base_network import BaseNetwork
from models.networks.loss import *
from models.networks.discriminator import *
from models.networks.generator import *
from models.networks.encoder import *




def find_network_using_name(target_network_name, filename):
    target_class_name = target_network_name + filename
    module_name = 'models.networks.' + filename

    print(f'module name that is being loaded is: {module_name}')
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    netG_cls = find_network_using_name(opt.netG, 'generator')
    parser = netG_cls.modify_commandline_options(parser, is_train)
    if is_train:
        netD_cls = find_network_using_name(opt.netD, 'discriminator')
        parser = netD_cls.modify_commandline_options(parser, is_train)
    netE_cls = find_network_using_name(opt.netE, 'encoder')
    parser = netE_cls.modify_commandline_options(parser, is_train)

    return parser


def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    print(f'this is the network')
    print(f'what  is cls and opt? {cls} {opt}')
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        net.cuda()
    else:
        net.cpu()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')

    return create_network(netG_cls, opt)


def define_D(opt):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt)


def define_E(opt):
    # there exists only one encoder type
    netE_cls = find_network_using_name(opt.netE, 'encoder')
    return create_network(netE_cls, opt)
