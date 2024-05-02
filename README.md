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

Reference: This project is based on a baseline model https://github.com/DCASE2023-Task7-Foley-Sound-Synthesis/dcase2023_task7_baseline
