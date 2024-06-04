"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import ResnetBlock3D as ResnetBlock3D
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from torchinfo import summary
### sina
# ref:"https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html"
# Set random seed for reproducibility
import random
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
### sina

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('few','normal', 'more', 'most', 'most512'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if self.opt.use_vae:
            # In case of VAE, we will sample from random z vector
            
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
            ### sina
        elif self.opt.use_noise:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
            ### sina
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
        if opt.num_upsampling_layers == 'most512':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
            self.up_5 = SPADEResnetBlock(nf // 2, nf // 4, opt)
            final_nc = nf // 4

        self.conv_img = nn.Sequential(
            nn.Conv2d(final_nc, opt.output_nc, 3, padding=1),  # sina changing the number of the channels for the output conv layer from 3 to opt.output_nc
            nn.Tanh()
        )

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'few':
            num_up_layers = 4
        elif opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        elif opt.num_upsampling_layers == 'most512':
            num_up_layers = 8

        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None, input_dist=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
            ### sina
        elif self.opt.use_noise:
            # print('yes got the noise')
            z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
            ### sina
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg, input_dist)

        if self.opt.num_upsampling_layers != 'few':
            
            x = self.up(x)
            x = self.G_middle_0(x, seg, input_dist)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg, input_dist)

        x = self.up(x)
        x = self.up_0(x, seg, input_dist)
        x = self.up(x)
        x = self.up_1(x, seg, input_dist)
        x = self.up(x)
        x = self.up_2(x, seg, input_dist)
        
        
        x = self.up(x)
        x = self.up_3(x, seg, input_dist)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)
        if self.opt.num_upsampling_layers == 'most512':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)
            x = self.up(x)
            x = self.up_5(x, seg, input_dist)


        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = nn.Tanh(x)

        return x


class SPADEEncGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('few','normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if self.opt.use_vae:
            # In case of VAE, we will sample from random z vector
            
            # self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        #FIXME: 
    
            self.fc = nn.Conv2d(self.opt.encoder_nc, 16 * nf, 3, padding=1) # encoder_nc should be the numbe of the channels for the encoder output 512?
            ### sina
        elif self.opt.use_noise:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
            ### sina
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
        if opt.num_upsampling_layers == 'most512':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            self.up_5 = SPADEResnetBlock(nf // 2, nf // 4, opt)
            final_nc = nf // 4


        self.conv_img = nn.Sequential(
            nn.Conv2d(final_nc, opt.output_nc, 3, padding=1),  # sina changing the number of the channels for the output conv layer from 3 to opt.output_nc
            nn.Tanh()
        )

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'few':
            num_up_layers = 4
        elif opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        elif opt.num_upsampling_layers == 'most512':
            num_up_layers = 8
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None, input_dist=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
            ### sina
        elif self.opt.use_noise:
            # print('yes got the noise')
            z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
            ### sina
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg, input_dist)

        if self.opt.num_upsampling_layers != 'few':
            
            x = self.up(x)
            x = self.G_middle_0(x, seg, input_dist)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most' or  self.opt.num_upsampling_layers == 'most512':
            x = self.up(x)

        x = self.G_middle_1(x, seg, input_dist)

        x = self.up(x)
        x = self.up_0(x, seg, input_dist)
        x = self.up(x)
        x = self.up_1(x, seg, input_dist)
        x = self.up(x)
        x = self.up_2(x, seg, input_dist)
        
        
        x = self.up(x)
        x = self.up_3(x, seg, input_dist)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = nn.Tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)







class StyleSPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        
        parser.add_argument('--num_upsampling_layers',
                            choices=('few','normal', 'more', 'most','most512'), default='few',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        
        # parser.add_argument('--resnet_n_downsample', type=int, default=5, help='number of downsampling layers in netG')
        # parser.add_argument('--resnet_n_blocks', type=int, default=2, help='number of residual blocks in the global generator network')
        # parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            # help='kernel size of the resnet block')
        # parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
        #                     help='kernel size of the first convolution')
        # parser.set_defaults(resnet_n_downsample=5)
        # parser.set_defaults(resnet_n_blocks=2)
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        # parser.set_defaults(norm_G='spectralspadeinstance3x3') dont use
        # parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        #print(f''using StyleSPADEGenerator')
        super().__init__()
        self.voxel_size = opt.voxel_size
        self.opt = opt
        nf = opt.ngf
        norm_layer_style = get_nonspade_norm_layer(opt, 'spectralsync_batch')
        # norm_layer_style = get_nonspade_norm_layer(opt, 'spectralinstance') dont use
        # norm_layer_style = get_nonspade_norm_layer(opt, opt.norm_E)
        
        activation = nn.ReLU(False)
        model = []

        ##   style encoder 

         # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer_style(nn.Conv2d(self.opt.output_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer_style(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2


        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer_style,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        self.model = nn.Sequential(*model)

        if self.opt.crop_size == 256:
            in_fea = 2 * 16
            self.opt.num_upsampling_layers = 'most'
        if self.opt.crop_size == 512:
            in_fea = 4 * 16
            self.opt.num_upsampling_layers = 'most512'
        if self.opt.crop_size == 128:
            in_fea = 1 * 16
            
        #print(f'variables for linear FC layer: in_fea: {in_fea}, nf: {nf}')
        
        self.fc_img = nn.Linear(in_fea * nf * 16 * 16, in_fea * nf //4)
        self.fc_img2 = nn.Linear(in_fea * nf // 4, in_fea * nf * 8 * 8,8)
        self.fc = nn.Conv2d(self.opt.semantic_nc, in_fea * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(in_fea * nf, in_fea * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(in_fea * nf, in_fea * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(in_fea * nf, in_fea * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        #Increase upsampling if you want the fake images to be generated in higher resolution -> 512
        if self.opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
        
        if self.opt.num_upsampling_layers == 'most512':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            self.up_5 = SPADEResnetBlock(nf // 2, nf // 4, opt)
            final_nc = nf // 4


        self.conv_img = nn.Sequential(
            nn.Conv2d(final_nc, opt.output_nc, 3, padding=1),  # sina changing the number of the channels for the output conv layer from 3 to opt.output_nc
            nn.Tanh()
        )

        self.up = nn.Upsample(scale_factor=2)


    def forward(self, input, image, input_dist=None):
        seg = input
        image = image

        #print(f'incoming image shape: {image.shape} and the seg shape {seg.shape}, ')
        if(self.voxel_size >1):
           
            x = self.model(image)
        #print(f'x shape after self.model(image): {x.shape}')
        if(self.opt.voxel_size >1):
            x = x.unsqueeze(2)  # Adds an extra dimension at the third position
            x = x.expand(-1, -1, 3, -1, -1)  # Expands the third dimension to a size of 3

        # if(self.voxel_size >1):
        #     x = x.reshape(x.size(0), -1, x.size(3), x.size(4))

        else:
            x = x.view(x.size(0), -1)
        #print(f'x shape before fc_img: {x.shape}')
        x = self.fc_img(x)
        x = self.fc_img2(x)

        #print(f'self.opt.crop_size {self.opt.crop_size}')
        if self.opt.crop_size == 256:
            in_fea = 2 * 16
        if self.opt.crop_size == 512:
            in_fea = 4 * 16
        if self.opt.crop_size == 128:
            in_fea = 1 * 16
    
        #Old try!!
        if(self.voxel_size >1):
            x = x.view(-1, in_fea * self.opt.ngf ,self.voxel_size, 8, 8)
        else:
            x = x.view(-1, in_fea * self.opt.ngf , 8, 8)
        #hard coded:
        #x= x.view(1,-1,8,8)
        #print(f'After view: {x.shape}')
        


        # print(f'in generator.py, after x.view(): {x.shape}')
        # print(f'seg.shape: {seg.shape}')
        # print(f'input_dist.shape: {input_dist.shape}')
        #print(f'x shape before failed line: {x.shape}')
        x = self.head_0(x, seg, input_dist)
        #print(f'After head_0: {x.shape}')

        # if self.opt.num_upsampling_layers != 'fgit adew':
            
        #     x = self.up(x)
        x = self.G_middle_0(x, seg, input_dist)
        #print(f'After G_middle_0: {x.shape}')


        # if self.opt.num_upsampling_layers == 'more' or \
        #    self.opt.num_upsampling_layers == 'most':
        #     x = self.up(x)

        # x = self.G_middle_1(x, seg, input_dist)

        x = self.up(x)
        x = self.up_0(x, seg, input_dist)
        x = self.up(x)
        x = self.up_1(x, seg, input_dist)
        x = self.up(x)
        x = self.up_2(x, seg, input_dist)
        x = self.up(x)
        x = self.up_3(x, seg, input_dist)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)
        if self.opt.num_upsampling_layers == 'most512':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)
            x = self.up(x)
            x = self.up_5(x, seg, input_dist)

 


        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = nn.Tanh(x)


        return x
    


class StyleSPADE3DGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--num_upsampling_layers',
                            choices=('few', 'normal', 'more', 'most', 'most512'), default='few',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):


        #print(f'runnning 3D SPADE gan!')
        super().__init__()
        self.voxel_size = opt.voxel_size
        self.opt = opt
        nf = opt.ngf
        norm_layer_style = get_nonspade_norm_layer(opt, 'spectralsync_batch')
        if self.opt.crop_size == 256:
            in_fea = 2 * 16
            self.opt.num_upsampling_layers = 'most'
            print(f'most upsampling layers')
        if self.opt.crop_size == 512:
            in_fea = 4 * 16
            self.opt.num_upsampling_layers = 'most512'
        if self.opt.crop_size == 128:
            in_fea = 1 * 16
        #in_fea: {in_fea}')
        activation = nn.ReLU(False)
        model = []

        ##   style encoder 

        #Padding needs different values for 3D
        in_kernel = opt.resnet_initial_kernel_size
        model += [nn.ReflectionPad3d((in_kernel//2, in_kernel//2, in_kernel//2, in_kernel//2, in_kernel//4, in_kernel//4)),

                  norm_layer_style(nn.Conv3d(self.opt.output_nc, opt.ngf,
                                       kernel_size=[in_kernel//3,in_kernel,in_kernel],
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            kernel_size_3d = [3,3,3]
            stride_3d = [2,2,2]
            model += [norm_layer_style(nn.Conv3d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size_3d, stride_3d, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks, first one takes 3D information
        for i in range(opt.resnet_n_blocks):
            #print(f'adding resnet block {i}')
            kernel_size_3d = [3,3,3]

            model += [ResnetBlock3D(opt.ngf * mult,  # use 3D version of ResnetBlock
                                  norm_layer=norm_layer_style,
                                  activation=activation,
                                  kernel_size=kernel_size_3d)]  

        self.model = nn.Sequential(*model)

        
        #hardcoded for testing, want to use with nf to make it more flexible
        self.fc_img = nn.Linear((in_fea *nf *27**2), (in_fea * nf *4))
        self.fc_img2 = nn.Linear(in_fea * nf *4 ,in_fea*nf*8*8*3 ) 

        self.fc = nn.Conv3d(self.opt.semantic_nc, in_fea * nf*2, 3, padding=1)


        self.head_0 = SPADEResnetBlock(in_fea * nf//8  *3 , in_fea * nf//3, opt)

        self.G_middle_0 = SPADEResnetBlock(85, in_fea * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(in_fea * nf, in_fea * nf, opt)


        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2* nf, opt)
        self.up_3 = SPADEResnetBlock(2* nf,  nf, opt)

        final_nc = nf

        #Increase upsampling if you want the fake images to be generated in higher resolution -> 512
        if self.opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(2 * nf, nf, opt)
            final_nc = nf // 2
        
        if self.opt.num_upsampling_layers == 'most512':
            self.up_4 = SPADEResnetBlock(4*nf, nf*2, opt)
            self.up_5 = SPADEResnetBlock(nf * 2,  nf//2, opt)
            final_nc = nf // 2


        self.conv_img = nn.Sequential(
            nn.Conv3d(final_nc, opt.output_nc, 3, padding=1),  
            nn.Tanh()
        )


    def forward(self, input, image, input_dist=None):
        #print(f'#############################')

        
        seg = input
        image = image
        nf = self.opt.ngf
        bs = self.opt.batchSize

        image = image.unsqueeze(1)
    
        depth= seg.shape[2]
        x = self.model(image)

        
        x = x.view(x.size(0), -1)
        x = self.fc_img(x)

        x = self.fc_img2(x)


        x = x.view(bs, -1, 8, 8, 8)  # reshaping to have depth dimension again
  
        x = self.head_0(x, seg, input_dist)

        x = self.G_middle_0(x, seg, input_dist)


        x = self.up(x)

        x = self.up_0(x, seg, input_dist)

        x = self.up(x)

        x = self.up_1(x, seg, input_dist)

        x = self.up(x)

        x = self.up_2(x, seg, input_dist)

        x = self.up(x)

        x = self.up_3(x, seg, input_dist)


        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)
        if self.opt.num_upsampling_layers == 'most512':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)
            x = self.up(x)
            x = self.up_5(x, seg, input_dist)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        

        return x




class SPADE3DGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('few', 'normal', 'more', 'most', 'most512'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf


        self.fc = nn.Conv3d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
        if opt.num_upsampling_layers == 'most512':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
            self.up_5 = SPADEResnetBlock(nf // 2, nf // 4, opt)
            final_nc = nf // 4

        self.conv_img = nn.Sequential(
            nn.Conv3d(final_nc, opt.output_nc, 3, padding=1),
            nn.Tanh()
        )

        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')





    def forward(self, input, input_dist=None):
        sw = self.opt.crop_size // (2 ** 5)
        sh = round(sw / self.opt.aspect_ratio)
        seg = input

        x = F.interpolate(seg, size=(sh, sh, sw))
        x = self.fc(x)
        x = self.head_0(x, seg, input_dist)

        if self.opt.num_upsampling_layers != 'few':
            x = self.up(x)
            x = self.G_middle_0(x, seg, input_dist)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg, input_dist)

        x = self.up(x)

        x = self.up_0(x, seg, input_dist)
        x = self.up(x)
        x = self.up_1(x, seg, input_dist)

        x = self.up(x)

        x = self.up_2(x, seg, input_dist)

        x = self.up(x)

        x = self.up_3(x, seg, input_dist)
       

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)
        if self.opt.num_upsampling_layers == 'most512':
            x = self.up(x)
            x = self.up_4(x, seg, input_dist)
            x = self.up(x)
            x = self.up_5(x, seg, input_dist)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        return x
