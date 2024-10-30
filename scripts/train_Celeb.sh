#!/usr/bin/env bash
DEVICE=0

# put your Celeb source directory path for the extracted faces and Dataframe and uncomment the following line
# FFPP_FACES_DIR=/your/dfdc/faces/directory
# FFPP_FACES_DF=/your/dfdc/faces/dataframe/path

# Celeb_SRC="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/Celeb-DF"
# VIDEODF_SRC="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/data/celebdf_videos.pkl"
# FACES_DST="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/FACES_DST"
# FACESDF_DST="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/FACESDF_DST"
# CHECKPOINT_DST="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/CHECKPOINT_DST"

# CELEB_FACES_DF="/path/to/your/celeb_faces_dataframe.pkl"
# CELEB_FACES_DIR="/path/to/your/celeb_faces_directory/"

# CELEB_FACES_DF = VIDEODF_SRC
# CELEB_FACES_DIR = FACES_DST

CELEB_FACES_DF="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/data/celebdf_videos.pkl"
CELEB_FACES_DIR="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/FACES_DST"


python train_binclass.py \
--net Xception \
--traindb celebdf \
--valdb celebdf \
--celeb_faces_df_path $CELEB_FACES_DF \
--celeb_faces_dir $CELEB_FACES_DIR \
--face scale \
--size 224 \
--batch 32 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 30000 \
--seed 41 \
--attention \
--device $DEVICE
