# frontal-face-trans
steps for operation
# training using CelebA dataset
1. python3 prepare_train.py
2. python3 generate_train.py
3. cd src/pix2pix
   copy traing and testing data to datasets folder
   python3 main.py --phase train
4. python3 main.py (--phase test)
# test for lfw dataset
1. python3 pre_process_lfw.py
2. python3 trans_lfw.py
3. python3 resize_lfw.py
4. cd src/pix2pix
   copy image data to datasets folder
   python3 main.py
# evaluate distance of CelebA testing data
1. python3 celeba_distance_calculate.py
# evaluate distance of lfw data
1. python3 lfw_distance_calculate.py
