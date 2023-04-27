**Tabular Diffusion**
This repository contains all code related to the 'Advancing Diffusion Models for Human Activity Recognition' MQP 2022-2023.
This directory contains all code related to the diffusion model and tabular diffusion model used in this project.

**Diffusion Resources**
- https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/#:~:text=Diffusion%20Models%20are%20generative%20models%2C%20meaning%20that%20they,recover%20the%20data%20by%20reversing%20this%20noising%20process.
- https://github.com/azad-academy/denoising-diffusion-model/blob/main/diffusion_model_demo.ipynb


**File Description**
- classifier.py contains the custom classifier. (In our evaluation method, we opted for scikit learn classifiers instead of our custom classifiers)
- ctgan_test.py contains the driver code for running CT-GAN model on UCI HAR dataset as part of our evaluation metrics.
- diffusion.py contains the training functions for the reverse diffusion steps.
- early_stopper.py contains the early stopping class used to stop the diffusion training.
- ema.py contains the exponential moving average class which is now deprecated from our project.
- evaluate.py contains the evaluation functions used for this project.
- extrasensory_test.py contains the driver code for running tabular diffusion model on toy data and experiment with Extrasensory dataset.
- gan.py contains the custom GAN class used in A term 2022 to gain an understanding into generative models.
- helper_plot.py contains the visualization class and plotting related code.
- model.py contains the diffusion model architectures class.
- uci_test.py contains the main driver class for this project including UCI HAR data preprocessing and diffusion model training.
- utils.py contains all the utility functions.

**Folder Description**
- /classifier_data contains the csv file generated when performing machine evaluation metrics.
- /classifier_models contains the saved custom classifier models.
- /diffusion_models contains the saved diffusion models.
- /figures contains all the saved visualization generated from evaluation metrics.
- /gan_models contains the saved custom GAN models.
- /practice contains the onboarding quizzes for G and Cindy in C term 2022
- /results contains some visualization generated for testing. (deprecated in favor of /figures directory)
- /training_loss contains some visualization on diffusion training loss. (deprecated in favor of /figures directory)
