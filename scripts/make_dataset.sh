#!/usr/bin/env bash
# This is a shebang (#!) followed by the path to the Bash shell interpreter. 
# It tells the system that this script should be run in a Bash shell environment.

#241029 12:12 make_dataset.sh runs successfully.

# echo ""
# echo "-------------------------------------------------"
# echo "| Index DFDC dataset                            |"
# echo "-------------------------------------------------"
# # put your dfdc source directory path and uncomment the following line
# # DFDC_SRC=/your/dfdc/train/split/source/directory
# python index_dfdc.py --source $DFDC_SRC

# echo ""
# echo "-------------------------------------------------"
# echo "| Index FF dataset                              |"
# echo "-------------------------------------------------"
# # put your ffpp source directory path and uncomment the following line
# # FFPP_SRC=/your/ffpp/source/directory
# python index_ffpp.py --source $FFPP_SRC

# SUJI
echo ""
export NO_ALBUMENTATIONS_UPDATE=1
echo "-------------------------------------------------"
echo "| Index Celeb-DF dataset                              |"
echo "-------------------------------------------------"
# put your ffpp source directory path and uncomment the following line
Celeb_SRC=/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/Celeb-DF

# /Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/PGM/DeepFakeDetection_icpr2020dfdc/.venv/bin/python index_celebdf.py --source $Celeb_SRC
/Users/suji/anaconda3/bin/python index_celebdf.py --source $Celeb_SRC 


# echo ""
# echo "-------------------------------------------------"
# echo "| Extract faces from DFDC                        |"
# echo "-------------------------------------------------"
# # put your source and destination directories and uncomment the following lines
# # DFDC_SRC=/your/dfdc/source/folder
# # VIDEODF_SRC=/previously/computed/index/path
# # FACES_DST=/faces/output/directory
# # FACESDF_DST=/faces/df/output/directory
# # CHECKPOINT_DST=/tmp/per/video/outputs
# python extract_faces.py \
# --source $DFDC_SRC \
# --videodf $VIDEODF_SRC \
# --facesfolder $FACES_DST \
# --facesdf $FACESDF_DST \
# --checkpoint $CHECKPOINT_DST

# echo ""
# echo "-------------------------------------------------"
# echo "| Extract faces from FF                         |"
# echo "-------------------------------------------------"
# # put your source and destination directories and uncomment the following lines
# # FFPP_SRC=/your/dfdc/source/folder
# # VIDEODF_SRC=/previously/computed/index/path
# # FACES_DST=/faces/output/directory
# # FACESDF_DST=/faces/df/output/directory
# # CHECKPOINT_DST=/tmp/per/video/outputs
# python extract_faces.py \
# --source $FFPP_SRC \
# --videodf $VIDEODF_SRC \
# --facesfolder $FACES_DST \
# --facesdf $FACESDF_DST \
# --checkpoint $CHECKPOINT_DST

echo ""
echo "-------------------------------------------------"
echo "| Extract faces from Celeb-DF                         |"
echo "-------------------------------------------------"
# put your source and destination directories and uncomment the following lines
Celeb_SRC="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/Celeb-DF"
VIDEODF_SRC="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/data/celebdf_videos.pkl"
FACES_DST="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/FACES_DST"
FACESDF_DST="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/FACESDF_DST"
CHECKPOINT_DST="/Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/CHECKPOINT_DST"
# python extract_faces.py \
# /Users/suji/Downloads/UTS_Masters/SEM3/IPPR_42177/PGM/DeepFakeDetection_icpr2020dfdc/.venv/bin/python extract_faces.py \
/Users/suji/anaconda3/bin/python extract_faces.py \
--source "$Celeb_SRC" \
--videodf "$VIDEODF_SRC" \
--facesfolder "$FACES_DST" \
--facesdf "$FACESDF_DST" \
--checkpoint "$CHECKPOINT_DST"

# --source $Celeb_SRC \
# --videodf $VIDEODF_SRC \
# --facesfolder $FACES_DST \
# --facesdf $FACESDF_DST \
# --checkpoint $CHECKPOINT_DST
