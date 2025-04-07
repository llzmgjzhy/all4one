
export CUDA_VISIBLE_DEVICES=0

seq_len=512
model=ALL4ONE
batch_size=128

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
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size $batch_size \
    --lradj 'COS' \
    --lr $lr \
    --epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0 \
    --enc_in 7 \
    --c_out 7 \
    --freq h \
    --img_width 256 \
    --img_height 256 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --llm_layer 6 \
    --llm_dim 3584 \
    --itr 1 \
    --model $model \
    --loss mse \
    --key_metric mse_loss \
    # --no_savemodel 
done
done
done
