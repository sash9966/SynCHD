"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE, SPADELight, SPADE3D


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        # create conv layers
        add_channels = 1 if (opt.norm_mode == 'clade' and not opt.no_instance) else 0
        if (opt.norm_mode == 'spade3d'): 
            self.conv_0 = nn.Conv3d(fin, fmiddle, kernel_size=3, padding=1)
            self.conv_1 = nn.Conv3d(fmiddle, fout, kernel_size=3, padding=1)
            
            if self.learned_shortcut:
                self.conv_s = nn.Conv3d(fin+add_channels, fout, kernel_size=1, bias=False)

        elif (opt.norm_mode == 'spade'):
            self.conv_0 = nn.Conv2d(fin+add_channels, fmiddle, kernel_size=3, padding=1)
            self.conv_1 = nn.Conv2d(fmiddle+add_channels, fout, kernel_size=3, padding=1)
            if self.learned_shortcut:
                self.conv_s = nn.Conv2d(fin+add_channels, fout, kernel_size=1, bias=False)
            


        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        if opt.norm_mode == 'spade':
            self.norm_0 = SPADE(spade_config_str, fin, opt.semantic_nc)
            self.norm_1 = SPADE(spade_config_str, fmiddle, opt.semantic_nc)
            if self.learned_shortcut:
                self.norm_s = SPADE(spade_config_str, fin, opt.semantic_nc)
        elif opt.norm_mode == 'clade':
            input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0)
            self.norm_0 = SPADELight(spade_config_str, fin, input_nc, opt.no_instance, opt.add_dist)
            self.norm_1 = SPADELight(spade_config_str, fmiddle, input_nc, opt.no_instance, opt.add_dist)
            if self.learned_shortcut:
                self.norm_s = SPADELight(spade_config_str, fin, input_nc, opt.no_instance, opt.add_dist)
        elif opt.norm_mode == 'spade3d':
            self.norm_0 = SPADE3D(spade_config_str, fin, opt.semantic_nc.opt.nhidden)
            self.norm_1 = SPADE3D(spade_config_str, fmiddle, opt.semantic_nc,opt.nhidden)
            if self.learned_shortcut:
                self.norm_s = SPADE3D(spade_config_str, fin, opt.semantic_nc,opt.nhidden)
        else:
            raise ValueError('%s is not a defined normalization method' % opt.norm_mode)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, input_dist=None):
        x_s = self.shortcut(x, seg, input_dist)
        #print(f'x: {x.shape}, seg: {seg.shape}, input_dist: {input_dist.shape}')
        #print(f'x_s: {x_s.shape}')

        #get info on the parameters of the network
        dx = self.conv_0(self.actvn(self.norm_0(x, seg, input_dist)))
        #print(f'dx after conv_0: {dx.shape}')
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg, input_dist)))
        #print(f'dx after_conv1: {dx.shape}')

        out = x_s + dx

        return out

    def shortcut(self, x, seg, input_dist=None):
        #print(f'shortcut is calledx: {x.shape}, seg: {seg.shape}')
        
        if self.learned_shortcut:
            #print(f'learned_shortcut is called')
            x_s = self.conv_s(self.norm_s(x, seg, input_dist))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out
    
class ResnetBlock3D(nn.Module):
    def __init__(self, dim, norm_layer,kernel_size,activation=nn.ReLU(False), ):
        super().__init__()

        pad = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2, (kernel_size[2] - 1) // 2)
        self.conv_block = nn.Sequential(
            nn.ConstantPad3d((pad[1], pad[1], pad[2], pad[2], pad[0], pad[0]), 0),
            norm_layer(nn.Conv3d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ConstantPad3d((pad[1], pad[1], pad[2], pad[2], pad[0], pad[0]), 0),
            norm_layer(nn.Conv3d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):

        #print(f'x: {x.shape}')
        y= self.conv_block(x)
        #print(f'y: {y.shape}')
        return x + y





# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()





        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        #print(f'X: {X.shape}')
        depth = X.shape[2]
        if (len(X.shape)== 4):
            X = X.unsqueeze(1)
        h_relu_sums = [0, 0, 0, 0, 0]  # to store sum of losses for each h_relu

        for z in range(X.shape[2]):  # loop over the z-axis (depth dimension)
            X_slice = X[:, :, z, :, :]
            if X_slice.shape[1] == 1:  # if number of channels is less than 3:
                X_slice = X_slice.repeat(1, 3, 1, 1)


                #TODO: Check if this vgg makes more senes!
            #X_slice = (X_slice + 1) / 2.0  # Adjust to [0, 1] range
            # Apply ImageNet normalization
            # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(X_slice.device)
            # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(X_slice.device)
            # X_slice = (X_slice - mean) / std

            h_relu1 = self.slice1(X_slice)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)
            h_relu_sums[0] += h_relu1.sum()
            h_relu_sums[1] += h_relu2.sum()
            h_relu_sums[2] += h_relu3.sum()
            h_relu_sums[3] += h_relu4.sum()
            h_relu_sums[4] += h_relu5.sum()

        h_relu_tensors = [torch.tensor(h_relu_sum) for h_relu_sum in h_relu_sums]

        ##:TODO: check if this is correct
        #h_relu_tensors = [torch.tensor(h_relu_sum)/depth for h_relu_sum in h_relu_sums]
        
        return h_relu_tensors   # return the output for each slice
    

#Unet from CHD paper -> use as a feature comparison, use pretrained unet from real images to compare features to synthetic

class Modified3DUNet(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter = 8):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='nearest')
        self.act = nn.Identity() if n_classes > 1 else nn.Sigmoid()

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter*2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter*4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*8)

        self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter*2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter*4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)




    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        #Advice from Phil: return second last encoder layer, emprical comparison of paper:
        #https://openreview.net/forum?id=AUiZyqYiGb , VGG also returns second to last layer!
        return context_4
    

        # # Level 5
        # out = self.conv3d_c5(out)
        # residual_5 = out
        # out = self.norm_lrelu_conv_c5(out)
        # out = self.dropout3d(out)
        # out = self.norm_lrelu_conv_c5(out)
        # out += residual_5
        # out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        # out = self.conv3d_l0(out)
        # out = self.inorm3d_l0(out)
        # out = self.lrelu(out)

        # # Level 1 localization pathway
        # out = torch.cat([out, context_4], dim=1)
        # out = self.conv_norm_lrelu_l1(out)
        # out = self.conv3d_l1(out)
        # out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # # Level 2 localization pathway
        # out = torch.cat([out, context_3], dim=1)
        # out = self.conv_norm_lrelu_l2(out)
        # ds2 = out
        # out = self.conv3d_l2(out)
        # out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # # Level 3 localization pathway
        # out = torch.cat([out, context_2], dim=1)
        # out = self.conv_norm_lrelu_l3(out)
        # ds3 = out
        # out = self.conv3d_l3(out)
        # out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # # Level 4 localization pathway
        # out = torch.cat([out, context_1], dim=1)
        # out = self.conv_norm_lrelu_l4(out)
        # out_pred = self.conv3d_l4(out)

        # ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        # ds1_ds2_sum_upscale = self.upsacle(ds2_1x1_conv)
        # ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        # ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        # ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle(ds1_ds2_sum_upscale_ds3_sum)

        # out_seg = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        