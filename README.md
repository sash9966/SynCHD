# Synthetic Image Generation to Aid Segmentation of Congenital Heart Disease Patient Images

Welcome to the GitHub repository dedicated to advancing the segmentation of congenital heart disease (CHD) patient images through synthetic image generation. This project aims to address the challenges posed by the rarity of CHD and the resultant data scarcity in machine learning applications, by generating synthetic yet anatomically accurate CT images of CHD patients.


## Branches Overview
- **Master**: This branch contains the fully functional 2D image generation architecture.
- **3dfinalgantesting**: Here, you'll find the working 3D architecture that extends the styleSPADE GANs into full 3D image generation. 

## Getting Started
To start training the models, simply run the `cmr_train.py` script. Ensure that the configuration files are set up according to your requirements for a seamless training experience. For the 3D case, indicate that the form is 3D with the argument --is_3D.
Check the train and base option in the options/ folder and adjust the paths necessary, like training data, reference images etc.


## Visual Demonstrations
We have included a GIF to visually demonstrate the masks:

![Masks Used in Image Generation](./gifs/real_image_axial_view.gif)

Thes GIF illustrate the masks used in our image generation process.
The masks were formulated with our own SDF4CHD method, outlined in the paper: 
 
SDF4CHD: Generative Modeling of Cardiac Anatomies with Congenital Heart Defects, https://arxiv.org/abs/2209.04223. 
Early results of this work for 2D generation are included in the experimental section as well.

Thank you for your interest in our project, and we hope our research contributes to significant advancements in the field of medical imaging and treatment of congenital heart disease.



This work builds on the work of Amirrajab et al. Please check out their github and their paper: https://github.com/sinaamirrajab/CardiacPathologySynthesis, https://arxiv.org/abs/2209.04223 and the SPADE GANs by Park et al. from Nvidia: https://arxiv.org/abs/1903.07291, https://github.com/nvlabs/spade/

Feel free to contact me for any questions: sastocke@stanford.edu
