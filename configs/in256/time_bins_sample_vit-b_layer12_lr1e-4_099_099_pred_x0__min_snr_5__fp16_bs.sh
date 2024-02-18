# set -ex

# pip install -r requirements.txt
# pip install -e .

# 修改 
USE_WANDB="True"
TIME_BINS="0 100, 500 600, 900 1000"

# DATA_DIR="../datasets/ILSVRC2012_ldm_256_diffuser/train/"
DATA_DIR="/data4/share/imagenet/train"

EXP_TIME_BINS="${TIME_BINS// /_}"
EXP_TIME_BINS="${EXP_TIME_BINS//,/}"

export CUDA_VISIBLE_DEVICES=1,3,4,5,6

# GPUS=$1
# BATCH_PER_GPU=$2
GPUS=4
BATCH_PER_GPU=128
EXP_NAME="time_bins_${EXP_TIME_BINS}_vit-b_layer12_lr1e-4_099_099_pred_x0__min_snr_5__fp16_bs${GPUS}x${BATCH_PER_GPU}"

MODEL_BLOB="/mnt/external"
if [ ! -d $MODEL_BLOB ]; then
    MODEL_BLOB="."
fi

OPENAI_LOGDIR="${MODEL_BLOB}/exp/guided_diffusion/$EXP_NAME"
# if permission denied
# sudo mkdir -p $OPENAI_LOGDIR && sudo chmod 777 $OPENAI_LOGDIR
mkdir -p $OPENAI_LOGDIR && chmod 777 $OPENAI_LOGDIR
OPENAI_LOGDIR=$OPENAI_LOGDIR \
    torchrun --nproc_per_node=${GPUS} --master_port=12457 scripts_vit/image_train_vit.py \
    --data_dir $DATA_DIR --image_size 32 --class_cond True --diffusion_steps 1000 \
    --noise_schedule cosine --rescale_learned_sigmas False \
    --lr 1e-4 --batch_size ${BATCH_PER_GPU} --log_interval 10 --beta1 0.99 --beta2 0.99 \
    --exp_name $EXP_NAME --use_fp16 True --weight_decay 0.03 \
    --use_wandb "$USE_WANDB" --model_name vit_base_patch2_32 --depth 12 \
    --predict_xstart True --warmup_steps 0 --lr_anneal_steps 0 \
    --mse_loss_weight_type min_snr_5 --clip_norm -1 \
    --in_chans 3 --drop_label_prob 0.15 --save_interval 1000 \
    --schedule_sampler min_max_bin_sampler --time_bins "$TIME_BINS"
    # --resume_checkpoint exp/guided_diffusion/debug_vit-b_layer12_lr1e-4_099_099_pred_x0__min_snr_5__fp16_bs3x64/model000440.pt \
