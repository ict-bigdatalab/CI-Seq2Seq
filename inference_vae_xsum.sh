DATA_NAME=xsum
DATA_DIR=data/${DATA_NAME}
MODEL_NAME=bart-large
CONFIG_DIR=model/${MODEL_NAME}
MODEL_DIR=output/${DATA_NAME}_${MODEL_NAME}
LDA_DICT=lda.dict
LDA_MODEL=lda.model

d_u=16
k_u=5
th=0.25
d_cc=128
d_sc=256
d_ss=128

sample_num=10
test_steps=40
wd=0.1
bs=2
lr=0.1

OUTPUT_DIR=output/${DATA_NAME}_${MODEL_NAME}/sp${sample_num}_ep${test_steps}_wd${wd}_bs${bs}_lr${lr}

CUDA_VISIBLE_DEVICES="" python run_bart_vae.py \
    --model_name_or_path ${MODEL_DIR} \
    --pretrained_model_dict_path ${CONFIG_DIR} \
    --lda_dict_path ${LDA_DICT} \
    --lda_model_path ${LDA_MODEL} \
    --config_name ${CONFIG_DIR} \
    --tokenizer_name ${CONFIG_DIR} \
    --validation_file ${DATA_DIR}/${DATA_NAME}_test.json \
    --text_column document \
    --summary_column summary \
    --max_source_length=512 \
    --max_target_length=64 \
    --per_device_eval_batch_size=$bs \
    --learning_rate_for_latent=$lr \
    --weight_decay_for_latent=$wd \
    --sample_num=$sample_num \
    --test_ep=$test_steps \
    --output_dir ${OUTPUT_DIR} \
    --predict_file ${OUTPUT_DIR}/text.json \
    --metric_file ${OUTPUT_DIR}/metric.json \
    --num_beams=6 \
    --seed=100 \
    --training_vae \
    --fuse_seq_info \
    --source_prefix "<cls>" \
    --aggregate "cls" \
    --causal_latent_size=$d_cc \
    --non_causal_latent_size=$d_sc \
    --style_size=$d_ss \
    --u_size=$d_u \
    --num_topics=$k_u \
    --tc_th=$th