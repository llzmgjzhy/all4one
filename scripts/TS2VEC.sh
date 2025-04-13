model=TS2VEC
batch_size=64
output_dim=1
epochs=200
lr=0.001
data=ETTh1

python ./src/main_ts2vec.py \
    --task forecast \
    --name "TS2VEC_ETTh1" \
    --model_name $model \
    --root_path ./src/datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --output_dir ./src/models/ts2vec/ \
    --data ETTh1 \
    --batch_size $batch_size \
    --lradj 'COS' \
    --lr $lr \
    --epochs $epochs \
    --dropout 0.1 \
    --output_dim $output_dim \
    --freq h \
    --itr 1 \
    --seed 2025 \