# 定义参数
$patch = 4
$stride = 2
$lr = 0.0001

# 循环
foreach ($patch in 4) {
    foreach ($stride in 2) {
        foreach ($lr in 0.0001) {

            python src/main_long_forecast.py `
                --task forecast `
                --output_dir ./src/experiments `
                --comment "forecasting using gpt2" `
                --seed 42 `
                --name "long forecasting ETTh1" `
                --records_file LongForecast_records.xls `
                --root_path ./src/datasets/ETT-small/ `
                --data_path ETTm1.csv `
                --data ETTh1 `
                --features M `
                --epochs 10 `
                --batch_size 256 `
                --lr $lr `
                --patch_size $patch `
                --stride $stride `
                --d_model 32 `
                --llm_dim 768 `
                --llm_layers 12 `
                --loss mse `
                --key_metric mse_loss `
                --model GPT4TS `
                --enc_in 7 `
                --d_ff 128 `
        
        }
    }
}
