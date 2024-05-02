Usage:

Download checkpoint here: https://drive.google.com/file/d/1hLbUi0veQ1D-yYGTxF-3rCfrVSIpzd6_/view?usp=sharing

Unzip the checkpoint and put the 'checkpoint' folder at the root level of this project.

1. Go to the project folder:
   cd dl4m-final
   
3. Install required packages:
   pip install -r requirements.txt
   
4. Train VQ-VAE:
   python train_vqvae.py
   
5. Extract code/embedding from trained VQ-VAE:
   python extract_code.py
   
6. Train PixelSnail:
   python train_pixelsnail.py
    
8. Inference:
   python inference.py
