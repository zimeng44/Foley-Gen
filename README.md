Introduction:

Foley-Gen is a generative machine learning model that generates noval foley sounds in 7 categories: Dog Bark, Footstep, Gunshot, Typing on Keyboard, Moving Motor Vehicle, Rain, Sneeze Cough. Each time the model runs, foley sounds in all 7 categories will be generated and saved to './synthesized'. The number of sounds that will be generated for each category can be modified by running inference.py with '--number_of_synthesized_sound_per_class = < number >'. The default number is 1 per category.

Usage:

If you want to use our checkpoint you can:

1. Download the checkpoint here: https://drive.google.com/file/d/1hLbUi0veQ1D-yYGTxF-3rCfrVSIpzd6_/view?usp=sharing

2. Unzip the checkpoint and put the 'checkpoint' folder at the root level of this project.

3. Run 'python inference.py'

The synthesized sound samples will be saved to `./synthesized`
   
============================================================================================================

If you want to train the models yourself:

1. Train VQ-VAE:
   python train_vqvae.py
   
2. Extract code/embedding from trained VQ-VAE:
   python extract_code.py
   
3. Train PixelSnail:
   python train_pixelsnail.py
    
4. Inference:
   python inference.py

The synthesized sound samples will be saved to `./synthesized`

Reference: This project is based on a baseline model https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline . Other models used in this project include MERT, VQ-VAE and PixelSnail.
