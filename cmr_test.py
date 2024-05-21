import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import numpy as np
import nibabel as nib
from util.filename import extract_filename
import SimpleITK as sitk
print(f'os.getcwd: {os.getcwd}')
ospath= os.getcwd()

if (ospath == "/home/sastocke/2Dslicesfor3D"):
    opt = TestOptions().parse()
    ref_img = sitk.ReadImage('/home/sastocke/data/testimages128/ct_1129_image.nii.gz')
    name = opt.name
    web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))


    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))



#Sherlock!
elif (ospath == "/scratch/users/sastocke/2Dslicesfor3D"):
    opt = TestOptions().parse()
    ref_img = sitk.ReadImage(opt.image_dir)
    opt.checkpoints_dir = "/scratch/users/sastocke/2Dslicesfor3D/checkpoints/"
    # opt.label_dir = "/scratch/users/fwkong/SharedData/Synthesized_correction_128"
    # opt.image_dir = "/scratch/users/sastocke/data/data/testresample128normed"
    #opt.results_dir = "/scratch/users/fwkong/SharedData/3dbatch2epoch100dataaug10x"
    name = opt.name


opt.batchSize = 1







dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results


#Registration try:


print(f'length of datalaoder: {len(dataloader)}')
# test
for i, data_i in enumerate(dataloader):

    print(f'i: {i}')
    generated = model(data_i, mode='inference').detach().cpu().numpy()
    
    img_path = data_i['path']

    print(f'shape of generated: {generated.shape}')
    img_numpy = generated[0,0,:,:,:]
    img_numpy_transposed = img_numpy.transpose(0,2,1)
    img = sitk.GetImageFromArray(img_numpy_transposed)
    img.CopyInformation(ref_img)
    path = data_i['gtname'][0]
    filename = os.path.basename(data_i['gtname'][0])
    sitk.WriteImage(img, os.path.join(opt.results_dir, filename))
    print(f'saved image: {filename}')

    
    
   

print(f'done')
    





if (ospath == "/home/sastocke/2Dslicesfor3D"):
    webpage.save()  
