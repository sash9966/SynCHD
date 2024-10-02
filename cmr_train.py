
import os
import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from data.base_dataset import repair_data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from util import html
from util.util import tensor2im, tensor2label
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# parse options
opt = TrainOptions().parse()
name_of_try = opt.name
#BUG: Unsure if for larger crop size this should be changed, seems to work without!
if opt.crop_size == 256:
     opt.resnet_n_downsample = 5
     opt.resnet_n_blocks = 2
else:
    opt.resnet_n_downsample = 4
    opt.resnet_n_blocks = 2
opt.use_vae = False


# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

    

for epoch in iter_counter.training_epochs():
    #print('epoch', epoch)
    iter_counter.record_epoch_start(epoch)

    

    #print(f'lenght of dataloader: {len(dataloader)}')
    for i, data_i in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}"), start=iter_counter.epoch_iter):
        

        #random int for 0-batchSize-1




        #First initalisation

        #look at data that's loaded:
        # print(f'i: {i}')
        #print(f' data_i: {data_i.keys()}')
        # print the value of the key with 'image'

        # data_i dict_keys(['label', 'image', 'instance', 'dist', 'path', 'gtname', 'index', 'segpair_slice'])
        # print(f' data_i: {data_i["image"]}')
        # print(f' image shape: {data_i["image"].shape}')
        # print(f'gt name is: {data_i["gtname"]}')
        # print(f'path is: {data_i["path"]}')

        #Set initial path:

        #When paths change, a new 3D volume is being generated

        iter_counter.record_one_iteration()

        #print(f'type of the image, is it PIL or torch tensor: {type(data_i['label'])}') 
        #print(f'data_i label type: {type(data_i)}')

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)



        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():


            #print shapes of the images being fed to visuals:
            # print(f' data_i[i] shape: {data_i["label"].shape}')
            # print(f' synthesized shape: {trainer.get_latest_generated().shape}')
            # print(f' data_i[image])]: {data_i["image"].shape}')
            synthetic=  trainer.get_latest_generated()


            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', synthetic),
                                   ('real_image', data_i['image'])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if (opt.batchSize >1):
                rand= np.random.randint(0, opt.batchSize-1)
            else:
                rand=0
            # print(f'shape of synthetic: {synthetic.shape}')
            # print(f'shape of data_i: {data_i["image"].shape}')
            # print(f'shape of label: {data_i["label"].shape}')
            # print(f'rand int is: {rand}')
            
            # #Print the unique values of the syntehtic iamge:
            # print(f'unique values of synthetic: {np.unique(synthetic.detach().cpu()[rand,0,:,:])}')


            # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # axs[0].imshow(data_i['label'].detach().cpu()[rand,0,:,:])
            # axs[0].axis('off')
            # axs[0].set_title('Input Label')
            # axs[1].imshow(synthetic.detach().cpu()[rand,0,:,:],cmap='gray')
            # axs[1].axis('off')
            # axs[1].set_title('Synthesized Image')
            # axs[2].imshow(data_i['image'].detach().cpu()[rand,0,:,:],cmap='gray')
            # axs[2].axis('off')
            # axs[2].set_title('Real Image')
            # plt.savefig(f'/scratch/users/sastocke/3dtrysherlock/2Dslicesfor3D/2Dslicesfor3D/checkpoints/{name_of_try}/web/images/epoch{epoch}_{i}_plotdepth.png')
            # fig.clf()

            #Save 3D stacked image


        #stack the images togehter to create a 3D image of the generated images



        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
        

        ##Failed 3D trial
        #     trainer.save('latest')
        # print(f'i : {i}')
        
        # if(i>0):
        #     print(f'current path is: {path} and data_i path is:{data_i["path"][0]}' )
        # if(i==0):
        #     path = data_i['path'][0]
        #     print(f'path : {path}')
        #     print(f'path type: {type(path)}')
        #     #Expected 3D
        #     image3D_epoch = torch.empty(221,512,512)
        #     print(f'initial done')
        #     print(f'image3D_epoch: {image3D_epoch.shape}')
        # elif((path != data_i['path'][0] ) and (iter_counter.needs_displaying) ):
        #     #save old 3D stacked, should be 221 images stacked together
        #     #Override for the new 3D stacked image
        #     affine = np.eye(4)
        #     image3D_epoch_np = image3D_epoch.detach().numpy()
        #     img = nib.Nifti1Image(image3D_epoch_np, affine)

        #     #get image nr. from path file name
        #     path = data_i['path'][0]
        #     print(f'path : {path}')
        #     print(f'path type: {type(path)}')
        #     imgNr= path[-17:-13]
        #     print(f'number {imgNr}')
        #     filename = "3Depoch{epoch}Image{imgNr}.nii.gz"
        #     nib.save(img, os.path.join(opt.checkpoints_dir, opt.name,'web','images', filename))

        #     # start new stacking for the next 3D image
        #     image3D_epoch = torch.empty(221,512,512)
        #     image3D_epoch[0,:,:] = trainer.get_latest_generated()[0,0,:,:]
        #     path = data_i['path']
        
        # else:
        #     print(f'image_3d_epoch shape= {image3D_epoch.shape}')
        #     print(f'latest generated is: {trainer.get_latest_generated().shape}')

        #     image3D_epoch[i%221,:,:] = trainer.get_latest_generated()[0,0,:,:]



    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()
    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)
    #check the shape of the stacked image
    #convert to numpy arry and save with nib
    




print('Training was successfully finished.')
