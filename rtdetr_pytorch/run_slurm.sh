# # Training USing A100 PRE INSTALLED DOCKER IMAGE FOR MMTRACKING ENVIRONMENT
srun -K --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=4 -p RTXA6000 --mem=100G \
--container-mounts=/dev/fuse:/dev/fuse,/home:/home,/netscratch:/netscratch,/ds-av:/ds-av,/ds:/ds,"`pwd`":"`pwd`" \
--container-image=/netscratch/palyakere/enroots/rtdetr_env.sqsh \
--container-workdir="`pwd`" \
--time=01-00:00:00 \
--job-name="bash" \
sh exec_script.sh

