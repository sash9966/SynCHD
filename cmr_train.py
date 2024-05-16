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

import os
from util import html
print(f'os.getcwd: {os.getcwd}')
ospath= os.getcwd()

if (ospath == "/home/sastocke/2Dslicesfor3D"):
    print(f'On local path!')
    opt = TrainOptions().parse()
    ref_img = sitk.ReadImage('/home/sastocke/data/testimages128/ct_1129_image.nii.gz')
    name_of_try = opt.name
    web_dir = os.path.join(opt.checkpoints_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))


    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))



#Sherlock!
elif (ospath == "/scratch/users/sastocke/3dtrysherlock/2Dslicesfor3D/2Dslicesfor3D"):
    opt = TrainOptions().parse()
    ref_img = sitk.ReadImage("/scratch/users/sastocke/data/data/resample128/images/ct_1001_image.nii.gz")
    opt.checkpoints_dir = "/scratch/users/sastocke/3dtrysherlock/2Dslicesfor3D/checkpoints"

    opt.label_dir = "/scratch/users/sastocke/data/data/resample128/masks"
    opt.image_dir = "/scratch/users/sastocke/data/data/resample128/images"
    name_of_try= opt.name
    web_dir = os.path.join(opt.checkpoints_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))

    opt.unet_path = "/scratch/users/fwkong/SharedData/UNets_correction/baseline/"
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
            print(f'image path: {data_i["path"][0]}')
            print(f'gt_label path: {data_i["gtname"][0]} ')
            
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
