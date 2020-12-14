# 11785-final-project: Self-learning for pedestrian detection in thermal images

Introduction
--
The repository contains two baseline model for fine-tuning Mask RCNN and Faster RCNN.

Dependency
--
The system and pre-trained models are developed based on torchvision.


Main scripts:
--
main_self_training.ipynb


Helper functions:
--
dataloader.py: functions to deal with the dataset, include train/val splitting, dataset, pseudo labeling, visualization tools

train_teacher.py: train a teacher model

self_training.py: using true labels and pseudo labels to train the model


Dataset and preparing
--
Download the FLIR thermal dataset: https://www.flir.com/oem/adas/adas-dataset-form/

The dataset has this structure:

├──FLIR_ADAS_1_3

|---├── train

|---|---├── Annotated_thermal_8_bit

|---|---|---├── images

|---|---├── RGB

|---|---|---├── images

|---|---├── thermal_8_bit

|---|---|---├── images

|---|---├── thermal_16_bit

|---|---|---├── images 

|---|---└── thermal_annotations.json

|---├── val

|---|---├── Annotated_thermal_8_bit

|---|---|---├── images

|---|---├── RGB

|---|---|---├── images

|---|---├── thermal_8_bit

|---|---|---├── images

|---|---├── thermal_16_bit

|---|---|---├── images

|---|---└── thermal_annotations.json

|---├── video

|---|---├── Annotated_thermal_8_bit

|---|---|---├── images

|---|---├── RGB

|---|---|---├── images

|---|---├── thermal_8_bit

|---|---|---├── images 

|---|---├── thermal_16_bit

|---|---|---├── images

|---|---└── thermal_annotations.json

|---└── ReadMe



We use the images in thermal_8_bit for training and testing. Run the train_vld_split() in the scripts will divide the train/thermal_annotations.json into train/train.json and train/vld.json. The test data and labels are saved in val/thermal_annotations.json. The train_vld_split() can run once (keep the the same splitting) or everytime (change the splitting everytime).

Before running the data, change the dataset_dir into the absolute path of FLIR_ADAS_1_3. That is, the folder "train" and "val" are saved in dataset_dir + "/train" and dataset_dir + "/val".

