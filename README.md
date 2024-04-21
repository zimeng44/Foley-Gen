Run these commands in the set order
  
3. python train_vqvae.py
4. python extract_code.py --vqvae_checkpoint 'vaqae-checkpoint-of-your-choice'
5. python train_pixelsnail.py
6. python inference.py --vqvae_checkpoint 'vaqae-checkpoint-of-your-choice' --pixelsnail_checkpoint 'pixelsnail-checkpoint-of-your-choice' --number_of_synthesized_sound_per_class 'intger-number-of-your-choice'
