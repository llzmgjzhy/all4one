# 定义参数
$patch = 4
$stride = 2
$lr = 0.0001

# 循环
foreach ($patch in 3) {
    foreach ($stride in 3) {
        foreach ($lr in 0.0001) {

            python src/main.py `
                --output_dir ./src/experiments `
                --comment "classification from Scratch" `
                --seed 3046 `
                --name VSB `
                --records_file Classification_records.xls `
                --data_dir ./src/datasets/VSB/train_preprocess_aligned.parquet `
                --meta_dir ./src/datasets/VSB/metadata_train.csv `
                --data_class VSB `
                --pattern TRAIN `
                --val_pattern TEST `
                --epochs 30 `
                --lr $lr `
                --patch_size $patch `
                --stride $stride `
                --optimizer RAdam `
                --d_model 768 `
                --pos_encoding learnable `
                --task classification `
                --pos_encoding learnable `
                --vsb_piece_dim 5000 `
                --loss cross_entropy `
                --key_metric mcc `
                --text_model GPT2 `
                --reduction_ratio 10 `
        
        }
    }
}
