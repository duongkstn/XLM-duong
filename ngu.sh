MLM:
python train.py --exp_name test_enzh_mlm --dump_path ./dumped/ --data_path ./data/processed/en-zh/ --lgs 'en-zh' --clm_steps '' --mlm_steps 'en,zh' --emb_dim 1024 --n_layers 6  --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size 32 --bptt 256 --optimizer adam,lr=0.0001  --epoch_size 200000 --validation_metrics _valid_mlm_ppl --stopping_criterion _valid_mlm_ppl,1

python train.py --exp_name test_enzh_mlm --dump_path ./dumped/ --data_path ./data/processed/en-zh/ --lgs 'en-zh' --clm_steps '' --mlm_steps 'en,zh' --emb_dim 64 --n_layers 2  --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size 256 --group_by_size true --bptt 96 --max_len 100 --optimizer adam,lr=0.0001  --epoch_size 1000 --max_epoch 1 --validation_metrics _valid_mlm_ppl --stopping_criterion _valid_mlm_ppl,4 --sinusoidal_embeddings true

kaggle
!ln -sf /kaggle/working/XLM-duong/data/processed/en-zh/valid.en-zh.en.pth /kaggle/working/XLM-duong/data/processed/en-zh/valid.en.pth
!ln -sf /kaggle/working/XLM-duong/data/processed/en-zh/valid.en-zh.zh.pth /kaggle/working/XLM-duong/data/processed/en-zh/valid.zh.pth
!ln -sf /kaggle/working/XLM-duong/data/processed/en-zh/test.en-zh.en.pth /kaggle/working/XLM-duong/data/processed/en-zh/test.en.pth
!ln -sf /kaggle/working/XLM-duong/data/processed/en-zh/test.en-zh.zh.pth /kaggle/working/XLM-duong/data/processed/en-zh/test.zh.pth
!python train.py --exp_name test_enzh_mlm --dump_path ./dumped --data_path ./data/processed/en-zh --lgs 'en-zh' --clm_steps '' --mlm_steps 'en,zh' --emb_dim 1024 --n_layers 2  --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size 256 --group_by_size true --bptt 96 --max_len 100 --optimizer adam,lr=0.0001  --epoch_size 498000 --max_epoch 7 --validation_metrics _valid_mlm_ppl --stopping_criterion _valid_mlm_ppl,4 --sinusoidal_embeddings true --local_rank 0


MLM + TLM:
python train.py --exp_name unsupMT_enzh --dump_path ./dumped/ --reload_model '/home/user/XLM-master/dumped/test_enzh_mlm/4znd93c8ap/checkpoint.pth,/home/user/XLM-master/dumped/test_enzh_mlm/4znd93c8ap/checkpoint.pth' --data_path ./data/processed/en-zh/ --lgs 'en-zh' --ae_steps 'en,zh' --bt_steps 'en-zh-en,zh-en-zh' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 64 --n_layers 2  --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size 256 --bptt 96 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size 2000 --eval_bleu true --stopping_criterion 'valid_en-zh_mt_bleu,10' --validation_metrics 'valid_en-zh_mt_bleu' --max_epoch 1