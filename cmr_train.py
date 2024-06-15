import sys
from options.train_options import TrainOptions
import data
from data.base_dataset import repair_data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
from util.util import plot_viewpoint_slices
import SimpleITK as sitk
from tqdm import tqdm
import torch
import os
from util import html

ospath= os.getcwd()

if (ospath == "/home/sastocke/2Dslicesfor3D"):
    print(f'On local path!')
    opt = TrainOptions().parse()
    ref_img = sitk.ReadImage('/home/sastocke/data/testimages128/ct_1001_image.nii.gz')
    name_of_try = opt.name
    web_dir = os.path.join(opt.checkpoints_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))


    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))


#Test git update...')
#Sherlock!
elif (ospath == "/scratch/users/sastocke/2Dslicesfor3D"):
    opt = TrainOptions().parse()
    ref_img = sitk.ReadImage("/scratch/users/fwkong/SharedData/imageCHDCleanedOriginal_aligned_all/aligned/normed_img128/ct_1001_image.nii.gz")
    name_of_try= opt.name
    web_dir = os.path.join(opt.checkpoints_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
    if (opt.unet_loss):
        opt.unet_path = "/scratch/users/fwkong/CHD/output/wh_raw_tests_cleanedall/UNets/aug7_real_3dunet/"
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))

    

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

###Testing dataugemntation scheme.... contrast drift may occur because of the data augmentation
def save_augmented_data(image, mask, save_dir, base_name, counter):
    os.makedirs(save_dir, exist_ok=True)
    
    image_path = os.path.join(save_dir, f"{base_name}_{counter}_image.nii.gz")
    mask_path = os.path.join(save_dir, f"{base_name}_{counter}_mask.nii.gz")
    
    # Convert tensors to numpy arrays
    image_np = image.detach().cpu().numpy().squeeze()
    mask_np = mask.detach().cpu().numpy().squeeze()
    
    # Save the image and mask using SimpleITK
    sitk_image = sitk.GetImageFromArray(image_np)
    sitk_mask = sitk.GetImageFromArray(mask_np)
    
    # Assuming the input images and masks are in the correct format
    sitk.WriteImage(sitk_image, image_path)
    sitk.WriteImage(sitk_mask, mask_path)
save_dir = '/path/to/testaugmentation'  # Provide the path to the folder where you want to save the augmented data
counter = 0 

for epoch in iter_counter.training_epochs():
    print('epoch', epoch)
    iter_counter.record_epoch_start(epoch)

    


    for i, data_i in enumerate(tqdm(dataloader, desc=f"Epoch {epoch} for {opt.name}, running on GPU: {opt.gpu_ids}"), start=iter_counter.epoch_iter):



        

        iter_counter.record_one_iteration()


        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)
        for j in range(real_image.shape[0]):  # Assuming batch dimension
            save_augmented_data(real_image[j], label[j], save_dir, "augmented_data", counter)
            counter += 1


        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            #show the 3D image
            latest_image = trainer.get_latest_generated()

            latest_image = latest_image.detach().cpu().numpy()
            real_image = (data_i['image']).detach().cpu().numpy()
            label = (data_i['label']).detach().cpu().numpy()


            plot_viewpoint_slices(label, latest_image, real_image,epoch,i,name_of_try,opt.checkpoints_dir)

                

            #sanity test:
            # print(f'image path: {data_i["path"][0]}')
            # print(f'gt_label path: {data_i["gtname"][0]} ')
            
            img = sitk.GetImageFromArray(latest_image[0,0,:,:,:])
            img.CopyInformation(ref_img)
            #sitk.WriteImage(img, f'{opt.checkpoints_dir}/{name_of_try}/web/images/latestsynthetic{epoch}.nii.gz')
            #Save 3D stacked image



        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')



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
