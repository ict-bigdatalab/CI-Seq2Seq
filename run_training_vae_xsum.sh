DATA_NAME=xsum
DATA_DIR=data/${DATA_NAME}
MODEL_NAME=bart-large
MODEL_DIR=model/${MODEL_NAME}
LDA_DICT=lda.dict
LDA_MODEL=lda.model

d_u=16
k_u=5
th=0.25
d_cc=128
d_sc=256
d_ss=128
lr=5e-5

OUTPUT_DIR=output/${DATA_NAME}_${MODEL_NAME}

CUDA_VISIBLE_DEVICES="" python run_bart_vae.py \
    --model_name_or_path ${MODEL_DIR} \
    --pretrained_model_dict_path ${MODEL_DIR} \
    --lda_dict_path ${LDA_DICT} \
    --lda_model_path ${LDA_MODEL} \
    --train_file ${DATA_DIR}/xsum_train.json \
    --validation_file ${DATA_DIR}/xsum_val.json \
    --text_column document \
    --summary_column summary \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=64 \
    --learning_rate=$lr \
    --max_source_length=512 \
    --max_target_length=64 \
    --num_train_epochs=15 \
    --output_dir ./${OUTPUT_DIR} \
    --model_save_dir ./${OUTPUT_DIR}/saved_model \
    --log_dir ./${OUTPUT_DIR}/log \
    --do_train \
    --seed=100 \
    --num_warmup_steps=800 \
    --weight_decay=0.01 \
    --kl_cost_annealing \
    --training_vae \
    --fuse_seq_info \
    --source_prefix "<cls>" \
    --aggregate "cls" \
    --causal_latent_size=$d_cc \
    --non_causal_latent_size=$d_sc \
    --style_size=$d_ss \
    --u_size=$d_u \
    --num_topics=$k_u \
    --tc_th=$th \
    --learning_rate_ratio=5.0 \
    --checkpoint_interval=10
