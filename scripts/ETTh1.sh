
export CUDA_VISIBLE_DEVICES=0

seq_len=512
# model=ALL4ONE
model=ALL4ONEFAST
# model=ALL4ONEonlyTS2VEC
batch_size=24
output_dim=1
epochs=10

for percent in 100
do
for pred_len in 96
do
for lr in 0.001
do

python ./src/main_long_forecast.py \
    --task forecast \
    --comment "forecasting using $model" \
    --name "longForecasting_ETTh1" \
    --root_path ./src/datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --output_dir ./src/experiments \
    --records_file LongForecast_record.xlsx \
    --data ETTh1 \
    --residual_path ./src/models/residual_projection/ETTh1/longForecasting_ETTh1_2025-04-19_15-58-16_6u8/checkpoints/model_best.pth \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size $batch_size \
    --lradj 'COS' \
    --lr $lr \
    --weight_decay 1e-5 \
    --epochs $epochs \
    --d_model 32 \
    --n_heads 8 \
    --d_ff 128 \
    --dropout 0.1 \
    --enc_in 7 \
    --c_out 7 \
    --val_interval 1 \
    --patience 2 \
    --output_dim $output_dim \
    --freq h \
    --img_width 256 \
    --img_height 256 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --llm_layer 6 \
    --llm_dim 3584 \
    --itr 1 \
    --model_name $model \
    --loss mse \
    --key_metric mse_loss \
    --seed 2025 \
    # --no_savemodel 
done
done
done
