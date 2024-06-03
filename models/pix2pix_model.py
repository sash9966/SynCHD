"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
try:
    from torch.cuda.amp import autocast as autocast, GradScaler
    AMP = True
except:
    AMP = False

class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        torch_version = torch.__version__.split('.')
        if int(torch_version[1]) >= 2:
            self.ByteTensor = torch.cuda.BoolTensor if self.use_gpu() \
                else torch.BoolTensor
        else:
            self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
                else torch.ByteTensor

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        self.amp = True if AMP and opt.use_amp and opt.isTrain else False

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            self.L1Loss = networks.L1Loss()
            if opt.unet_loss: 
                self.criterionUnet = networks.Modified3DUNetLoss(in_channels=1, n_classes=8, base_n_filter=16, gpu_ids= self.opt.gpu_ids, pretrained_model_path=self.opt.unet_path)

            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss() 
                self.L1Loss = networks.L1Loss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_semantics, real_image, input_dist = self.preprocess_input(data)
        
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, input_dist)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, input_dist)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar, xout = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                #print(f'using generate_fake')
                fake_image, _ , _ = self.generate_fake(input_semantics, real_image, input_dist)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            #print(f'Train is False or continue train is True')
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def voxel_semantics(self,input_label, label_map):
        input_semantics_slices = []
        #print(f'input_label shape: {input_label.shape}')
        for i in range(input_label.shape[-1]):
            # Get the slice
            #print(f'input_label shape: {input_label.shape}')
            #print(f'i: {i}')
            
            input_label_slice = input_label[..., i]
            label_map_slice = label_map[..., i].unsqueeze(1)

            #print(f'input_label_slice shape: {input_label_slice.shape}')
            #print(f'label_map_slice shape: {label_map_slice.shape}')


            # Perform the scatter operation on the slice
            input_semantics_slice = input_label_slice.scatter_(1, label_map_slice.clamp(max=7), 1.0)
            input_semantics_slice = input_semantics_slice.unsqueeze(-1)
            #print(f'input_semantics_slice shape: {input_semantics_slice.shape}')

            # Append to the list
            input_semantics_slices.append(input_semantics_slice)

        
        # Concatenate the list into a tensor
        input_semantics = torch.cat(input_semantics_slices, dim=-1)
        #print(f'input_semantics shape after voxel_cat: {input_semantics.shape} ')

        return input_semantics
    


    def voxel_semantics_3D(self,input_label, label_map):
        # Initialize an empty list to store the processed 3D slices
        input_semantics_volumes = []
        torch.cuda.synchronize()


        # Loop through the 3D volume
        for i in range(input_label.shape[-1]):
            # Get the 3D slice
            #print(f'i: {i} ')
            input_label_volume = input_label[..., i]
            #print(f'input_label_volume shape: {input_label_volume.shape}')
            label_map_volume = label_map[..., i].unsqueeze(1)


            input_semantics_volume = input_label_volume.scatter_(1, label_map_volume.clamp(max=7), 1.0)
            #print(f'input_semantics_volume shape after scatter: {input_semantics_volume.shape}')
            # Add an extra dimension for concatenation
            input_semantics_volume = input_semantics_volume.unsqueeze(-1)

            # Append to the list
            input_semantics_volumes.append(input_semantics_volume)

        # Concatenate the list into a tensor
        input_semantics = torch.cat(input_semantics_volumes, dim=-1)

        return input_semantics




    def preprocess_input(self, data):
        # move to GPU and change data types

        data['label'] = data['label'].long()
        if self.use_gpu():
            if(torch.cuda.is_available()):
                data['label'] = data['label'].cuda()
                data['instance'] = data['instance'].cuda()
                data['image'] = data['image'].cuda()
                data['dist'] = data['dist'].cuda()
            else:
                data['label'] = data['label'].cpu()
                data['instance'] = data['instance'].cpu()
                data['image'] = data['image'].cpu()
                data['dist'] = data['dist'].cpu()

        
        # create one-hot label map
        label_map = data['label']





        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        
        #print(f'label_map size: {label_map.size()}')
        if(self.opt.voxel_size == 0):
            bs, _, h, w, = label_map.size()
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
        elif(self.opt.voxel_size>= 1):
            bs,h,w,depth = label_map.size()
            input_label = self.FloatTensor(bs, nc, h, w,depth).zero_()



        if(self.opt.voxel_size >= 1):
            if(self.opt.is_3D):
                input_semantics = self.voxel_semantics_3D(input_label,label_map)
            else:
                input_semantics = self.voxel_semantics(input_label,label_map)
        else:
            input_semantics = input_label.scatter_(1, label_map.clamp(max=7), 1.0)
        
        if self.opt.no_BG:
            input_semantics[:,0,:,:]= 0

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image'], data['dist']

    def compute_generator_loss(self, input_semantics, real_image, input_dist):
        G_losses = {}

        fake_image, KLD_loss, L1_loss = self.generate_fake(
            input_semantics, real_image, input_dist, compute_kld_loss=self.opt.use_vae)

        G_losses['L1'] = L1_loss
        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss
            G_losses['L1'] = L1_loss
        # print('size of fake image is ', fake_image.shape, 'size of the real image is ', real_image.shape)

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if self.opt.unet_loss:
            bs = self.opt.batchSize
            unet_loss =0
            for i in range(bs):
                fake_img_i = fake_image[i,:,:,:,:].unsqueeze(0)
                real_img_i = real_image[i,:,:,:].unsqueeze(0)

                unet_loss += self.criterionUnet(fake_img_i,real_img_i) * self.opt.lambda_unet
                G_losses ['UNET'] = unet_loss/bs


        if not self.opt.no_vgg_loss:
            if self.amp:
                with autocast():
                    G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg
            else:
                G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, input_dist):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ , _ = self.generate_fake(input_semantics, real_image, input_dist)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        if self.amp:
            with autocast():
                mu, logvar, xout = self.netE(real_image)
        else:
            mu, logvar, xout = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, xout

    def generate_fake(self, input_semantics, real_image, input_dist, compute_kld_loss=False):
        z = None
        KLD_loss = None
        L1_loss = None
        if self.opt.use_vae:
            print(f'using VAE')
            z, mu, logvar, xout = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld
                

        if self.amp:
            with autocast():
                fake_image = self.netG(input_semantics, z=z, input_dist=input_dist)
        if self.opt.netG== 'stylespade' or self.opt.netG== 'stylespade3d':
            

            fake_image = self.netG(input_semantics, real_image, input_dist=input_dist)
        elif self.opt.netG == 'spade3d':
            fake_image = self.netG(input_semantics, input_dist=input_dist)
            if self.opt.phase == 'train':
                L1_loss = self.L1Loss(fake_image[0,:,:,:], real_image ) * self.opt.lambda_L1


        else:
            fake_image = self.netG(input_semantics, z=z, input_dist=input_dist)
        
        if self.opt.use_vae:
            
            if compute_kld_loss:
                #permute to the same shape as the fake image that's generated 
                real_image = real_image.permute(0, 3, 1, 2)
                L1_loss = self.L1Loss(fake_image, real_image ) * self.opt.lambda_L1

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss, L1_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        #check out size and shape of the images
        vs = self.opt.voxel_size
        size = self.opt.crop_size
        
        if(self.opt.voxel_size > 0 and not self.opt.is_3D):
            real_image = real_image.unsqueeze(0)

            ##BUG: for some reason cat threw dimension error even though shapes were the same. Fix with reshaping..
            input_semantics = input_semantics.view(self.opt.batchSize, 8, vs, size, size)
            fake_image = fake_image.view(self.opt.batchSize, 1, vs, size, size)
            real_image = real_image.view(self.opt.batchSize, 1, vs, size, size)
            # Fake has dim: [batch_size, channel, depth, height, width] no need for batch size
            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)
        elif(self.opt.is_3D):
            input_semantics = input_semantics.view(self.opt.batchSize, 8, size, size, size)
            fake_image = fake_image.view(self.opt.batchSize, 1, size, size, size)
            real_image = real_image.view(self.opt.batchSize, 1, size, size, size)

            fake_concat = torch.cat([input_semantics, fake_image], dim=1)
            real_concat = torch.cat([input_semantics, real_image], dim=1)
        
        
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)


        if self.amp:
            with autocast():
                discriminator_out = self.netD(fake_and_real)
        else:
            discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
