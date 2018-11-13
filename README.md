# frontal-face-trans
steps:
# training using CelebA dataset
1. python3 prepare_train.py
2. python3 generate_train.py
3. cd src/pix2pix
   python3 main.py --phase train
4. python3 main.py (--phase test)
# test for lfw dataset
