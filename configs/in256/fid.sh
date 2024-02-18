
# pip install -r requirements.txt
# pip install -e .

if [ ! -d edm ]; then
    git clone https://github.com/NVlabs/edm.git
fi

export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=1,3,4,5,6

GPUS=1
IMG_SIZE=32
BATCH_SIZE=32
NUM_SAMPLES=50000
MODEL_NAME="vit_xl_patch2_32"
DEPTH=28
GUIDANCE_SCALES="1.5"
STEPS="50"
PRED_X0=True


# ----------- scale loop ------------- #
for GUIDANCE_SCALE in $GUIDANCE_SCALES
do

for STEP in $STEPS
do

# OPENAI_LOGDIR="exp/guided_diffusion/xl_samples${NUM_SAMPLES}_step${STEP}_scale${GUIDANCE_SCALE}/"
OPENAI_LOGDIR="../exp/guided_diffusion/xl_samples50000_step50_scale1.5/"

cd edm
torchrun --standalone --nproc_per_node=$GPUS fid.py calc --images=$OPENAI_LOGDIR --ref=https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz --num $NUM_SAMPLES 
cd ..

done
done
# ----------- scale loop ------------- #

echo "----> DONE <----"


# -----------------------------------
#          expected output
# -----------------------------------
# Calculating FID...
# 2.0559