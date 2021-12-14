# CMSE822_ObjectDetection
Object Detection Project for CMSE 822 - Parallel Computing

Needs files for the COCO dataset in the following structure:
-root
 -annotations
   -instances_train2017.json
   -instances_val2017.json
 -images
   -train2017
   -val2017
   -test2017
  
For Exploratory Data Analysis check
EDA_COCO_Dataset.ipynb

To train launch 
python train.py --data-path path-to-root 

To launch training in parallel 
python -u -m torch.distributed.launch --nproc_per_node=# train_parallel.py

To test 
python test_image.py --input path-to-test-image --pretrained-model path-to-checkpoint-file



This implementation borrows a lot of parts from NVIDIA/DeepLearningExamples github repo
