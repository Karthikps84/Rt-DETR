#We do not need dependencies becase the image has everything now
### Dependencies and compilation scripts
#sh compile.sh
## run training
echo 'starting the execution of the script now'

#exdark
#echo 'Mounting ExDark Data right now'


#coco
#echo 'Mounting Coco Data right now'

#Exdark
datapipefs /home/palyakere/Datasets/exdark/mount_data --archives "/netscratch/hashmi/dataset_annotations/exdark.zip"

#screen -r 2501642.interactive
#Coco
#datapipefs /home/palyakere/Datasets/coco/mount_data --archives "/ds-av/public_datasets/coco/original/train2017.zip"
#datapipefs /home/palyakere/Datasets/coco/val --archives "/ds-av/public_datasets/coco/original/val2017.zip"

pip install mmengine
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0
#python tools/train.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml
#python tools/train.py -c configs/rtdetr/enhancer/llrtdetr_r18vd_6x_exdark.yml -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth

python tools/train.py -c configs/rtdetr/enhancer/llrtdetr_r18vd_6x_exdark.yml -r /netscratch/palyakere/work_dirs/RTDETR/R18/exdark/with_torch/retrain/checkpoint0019.pth

#python tools/train.py -c configs/rtdetr/enhancer/llrtdetr_r18vd_6x_darkface.yml -t https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth
#python tools/train.py -c configs/rtdetr/enhancer/llrtdetr_r18vd_6x_coco.yml