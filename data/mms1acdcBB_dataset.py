"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset



import os
import nibabel as nib
import util.cmr_dataloader as cmr
import util.cmr_transform as cmr_tran
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import glob


TR_CLASS_MAP_MMS_SRS= {'LV': 0,'RV': 1, 'LA': 2,'RA': 3, 'Myo' : 4, 'Aorta':5,'Pulminary' : 6, 'Background':7 }
TR_CLASS_MAP_MMS_DES= {'LV': 0,'RV': 1, 'LA': 2,'RA': 3, 'Myo' : 4, 'Aorta':5,'Pulminary' : 6, 'Background':7 }

class Mms1acdcBBDataset(BaseDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
       
        # parser.set_defaults(label_nc=4)

        # parser.set_defaults(crop_size=128)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(add_dist=False)
        
        #For training
        # parser.add_argument('--label_dir', type=str, required=False, default = "/home/sastocke/data/alltrainingdata/data/segmentation",
        #                     help='path to the directory that contains label images')
        # parser.add_argument('--image_dir', type=str, required=False, default ="/home/sastocke/data/alltrainingdata/data/images" ,
        #                     help='path to the directory that contains photo images')
        # parser.add_argument('--label_dir', type=str, required=False, default = "/scratch/users/sastocke/data/data/segmentation",
        #                     help='path to the directory that contains label images')
        # parser.add_argument('--image_dir', type=str, required=False, default ="/scratch/users/sastocke/data/data/images" ,
        #                     help='path to the directory that contains photo images')
        

       #For testing
        # parser.add_argument('--label_dir', type=str, required=False, default = "/home/sastocke/data/testmasks128/",
        #                     help='path to the directory that contains label images')
        # parser.add_argument('--image_dir', type=str, required=False, default ="/home/sastocke/data/testimages128" ,
        #                      help='path to the directory that contains photo images')
        # parser.add_argument('--label_dir', type=str, required=False, default = "/home/sastocke/2Dslicesfor3D/data/images/",
        #                     help='path to the directory that contains label images')
        # parser.add_argument('--image_dir', type=str, required=False, default ="/home/sastocke/2Dslicesfor3D/data/masks/" ,
        #                     help='path to the directory that contains photo images')
        
        parser.add_argument('--label_dir', type=str, required=False, default = "/scratch/users/fwkong/SharedData/imageCHDCleanedOriginal_aligned_all/aligned/seg_nii_gz_only_128",
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=False, default ="/scratch/users/fwkong/SharedData/imageCHDCleanedOriginal_aligned_all/aligned/normed_img128",
                            help='path to the directory that contains photo images')
        
        # parser.add_argument('--label_dir_B', type=str, required=False, default = "/Users/saschastocker/Desktop/Data/StyleTransfer/segmentationTestFullResolution",
        #                     help='path to the directory that contains label images')
        

        # parser.add_argument('--image_dir_B', type=str, required=False, default ="/Users/saschastocker/Desktop/Data/StyleTransfer/imageTestFullResolution" ,
        #                     help='path to the directory that contains photo images')
        # parser.add_argument('--instance_dir', type=str, default='',
        #                     help='path to the directory that contains instance maps. Leave black if not exists')
        # parser.add_argument('--acdc_dir', type=str, required=False, default = "/Users/saschastocker/Desktop/Data/StyleTransfer/SlicedMRI/patient101_frame01.nii/",
        #                     help='path to the directory that contains label images')
                        
        return parser

    def get_paths(self, opt):
        """
        To prepare and get the list of files
        """

        print(f'opt.label_dir: {opt.label_dir}')
        print(f'opt.image_dir: {opt.image_dir}')
        SA_mask_list_all = sorted(glob.glob(os.path.join(opt.label_dir, '*.nii.gz')))
        # check if exists:
        SA_mask_list = []
        for s in SA_mask_list_all:
            if not os.path.exists(os.path.join(opt.results_dir, os.path.basename(s))):
                SA_mask_list.append(s)

        if(opt.phase == 'test'):
            #For test we will generate images with different mask but paired with one patient image for the background.
            single_image = opt.image_dir
            SA_image_list = [single_image] * len(SA_mask_list)
            print(f'length of SA_image_list: {len(SA_image_list)}')
            print(f'length of SA_mask_list: {len(SA_mask_list)}')
        else:
            SA_image_list = sorted(os.listdir(os.path.join(opt.image_dir)))

        # SA_image_list_B = sorted(os.listdir(os.path.join(opt.image_dir_B)))
        # SA_mask_list_B = sorted(os.listdir(os.path.join(opt.label_dir_B)))

        #pathologies = sorted(os.listdir(os.path.join(opt.acdc_dir)))
        
        print(f'length of SA_image_list: {len(SA_image_list)}')
        print(f'length of SA_mask_list: {len(SA_mask_list)}')




        # assert len(SA_mask_list_B) == len(SA_image_list_B) 
        if(opt.phase != 'test'):

            assert len(SA_image_list) == len(SA_mask_list)


        SA_filename_pairs = [] 
        for i in range(len(SA_image_list)):
            #add *10 because of random data augmentation, should generate more data!
            if opt.phase == 'train':
                SA_filename_pairs += [(os.path.join(opt.image_dir,SA_image_list[i]), os.path.join(opt.label_dir, SA_mask_list[i]))] * opt.multi_data
            elif opt.phase == 'test':
                SA_filename_pairs += [(os.path.join(opt.image_dir,SA_image_list[i]), os.path.join(opt.label_dir, SA_mask_list[i]))]

            else:
                print(f'phase is not set to train or test!')
                break
        
        self.img_list = SA_image_list
        self.msk_list = SA_mask_list
        self.filename_pairs = SA_filename_pairs


        print(f'len of filename pairs: {len(self.filename_pairs)}')
        return self.filename_pairs, self.img_list, self.msk_list



    def initialize(self, opt):
        self.opt = opt
        #print(f'filename pairs trying to be read from options: {self.opt}')
        self.filename_pairs, _, _  = self.get_paths(self.opt)
        print(f'opt : {opt}')


   

        if opt.isTrain:
            train_transforms = Compose([
                # cmr_tran.Resample(self.opt.target_res,self.opt.target_res), #1.33
                # cmr_tran.CenterCrop2D((self.opt.crop_size,self.opt.crop_size)),
                
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                #cmr_tran.RandomRotation90(p=0.7),
                
                cmr_tran.ToTensor(),
                cmr_tran.NormalizeMinMaxpercentile3D(range=(-1,1), percentiles=(1,99)),
                # cmr_tran.NormalizeLabel(),
                # cmr_tran.NormalizeMinMaxRange(range=(-1,1)),
                
                # cmr_tran.PercentileBasedRescaling(out_min_max=(-1,1), percentiles=(1,99)),  #TODO: make sure the normalization is performed on the volume data not slice-by-slice
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=1),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.ClipNormalize(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipZscoreMinMax(min_intensity= 0, max_intensity=4000),
                cmr_tran.DataAugmentation3D(opt),
                # cmr_tran.RandomHorizontalFlip2D(p=0.7),
                # cmr_tran.RandomVerticalFlip2D(p=0.7),
                cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        else:
            train_transforms = Compose([
                # cmr_tran.Resample(self.opt.target_res,self.opt.target_res), #1.33
                # cmr_tran.CenterCrop2D((self.opt.crop_size,self.opt.crop_size)),
                # cmr_tran.RandomDilation_label_only(kernel_shape ='elliptical', kernel_size = 3, iteration_range = (1,2) , p=0.5),
                # cmr_tran.RandomRotation(degrees=90),
                # cmr_tran.RandomRotation(p=0.5),
                
                cmr_tran.ToTensor(),
                #cmr_tran.NormalizeInstance3D(range=(-1,1), percentiles=(1,99)),
                # cmr_tran.NormalizeMinMaxRange(range=(-1,1)),
                

                # cmr_tran.PercentileBasedRescaling(out_min_max=(-1,1), percentiles=(1,99)),  #TODO: make sure the normalization is performed on the volume data not slice-by-slice
                # cmr_tran.RandomElasticTorchio_label_only(num_control_points  = (8, 8, 4), max_displacement  = (14, 14, 1), p=1),
                
                # cmr_tran.RandomElasticTorchio(num_control_points  = (8, 8, 4), max_displacement  = (20, 20, 0), p=0.5),
                # cmr_tran.ClipScaleRange(min_intensity= 0, max_intensity=4000),
                # cmr_tran.ClipTanh(),
                # cmr_tran.ClipScaleRange(),
                # cmr_tran.RandomHorizontalFlip2D(p=0.5),
                # cmr_tran.RandomVerticalFlip2D(p=0.5),
                cmr_tran.UpdateLabels(source=TR_CLASS_MAP_MMS_SRS, destination=TR_CLASS_MAP_MMS_DES)

            ])
        
        #if(opt.phase == 'test'):
            #self.cmr_dataset(cmr.MRI2DSegmentationDataset(self.msk_list, transform = train_transforms, slice_axis=2, canonical = False))

        #self.cmr_dataset = cmr.MRI2DSegmentationDataset(self.filename_pairs,voxel_size = opt.voxel_size, transform = train_transforms, slice_axis=2,  canonical = False)
        self.cmr_dataset = cmr.MRI3DSegmentationDataset(self.filename_pairs, transform = train_transforms,  canonical = False)
        print(f'opt voxel size: {opt.voxel_size}, type: {type(opt.voxel_size)}, ')

        
        
        size = len(self.cmr_dataset)
        self.dataset_size = size


    def __getitem__(self, index):
        # Label Image
        data_input = self.cmr_dataset[index]
        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_tensor = data_input["gt"] # the label map equals the instance map for this dataset
        if not self.opt.add_dist:
            dist_tensor = 0

        #3D with full volumes
        if self.opt.is_3D:
                    input_dict = {'label': data_input['gt'],
                    'image': data_input['input'],
                    'instance': instance_tensor,
                    'dist': dist_tensor,
                    'path': data_input['filename'],
                    'gtname': data_input['gtname'],
                    'index': data_input['index'],

                    }

        #2D with slices
        else:
            input_dict = {'label': data_input['gt'],
                        'image': data_input['input'],
                        'instance': instance_tensor,
                        'dist': dist_tensor,
                        'path': data_input['filename'],
                        'gtname': data_input['gtname'],
                        'index': data_input['index'],
                        'segpair_slice': data_input['segpair_slice'],
                        }

        return input_dict
    
    def __len__(self):
        return self.cmr_dataset.__len__()