#!/bin/sh -x

## Grocery_and_Gourmet_Food
#python main.py --model_name BPR --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'
#
#python main.py --model_name NCF --emb_size 64 --layers '[64]' --lr 5e-4 --l2 1e-7 --dropout 0.2 --dataset 'Grocery_and_Gourmet_Food'
#
#python main.py --model_name Tensor --emb_size 64 --lr 1e-4 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'
#
#python main.py --model_name GRU4Rec --emb_size 64 --hidden_size 100 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
#
#python main.py --model_name NARM --emb_size 64 --hidden_size 100 --attention_size 16 --lr 1e-3 --l2 1e-4 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
#
#python main.py --model_name SASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
#
#python main.py --model_name TiSASRec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
#
#python main.py --model_name CFKG --emb_size 64 --margin 1 --include_attr 0 --lr 1e-4 --l2 1e-8 --dataset 'Grocery_and_Gourmet_Food'
#
#python main.py --model_name SLRCPlus --emb_size 64 --lr 5e-4 --l2 1e-5 --check_epoch 10 --dataset 'Grocery_and_Gourmet_Food'
#
#python main.py --model_name Chorus --emb_size 64 --margin 1 --lr 5e-4 --l2 1e-5 --check_epoch 10 --epoch 50 --early_stop 0 --batch_size 512 --dataset 'Grocery_and_Gourmet_Food' --stage 1
#python main.py --model_name Chorus --emb_size 64 --margin 1 --lr_scale 0.1 --lr 1e-3 --l2 0 --check_epoch 10 --dataset 'Grocery_and_Gourmet_Food' --base_method 'BPR' --stage 2
#
#python main.py --model_name KDA --emb_size 64 --include_attr 1 --freq_rand 0 --lr 1e-3 --l2 1e-6 --num_heads 4 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
#
#python main.py --model_name FourierTA --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
#
#python main.py --model_name SRGNN --emb_size 64 --num_layers 1 --lr 1e-4 --l2 1e-6 --history_max 20 --dataset 'Grocery_and_Gourmet_Food'


# Beauty
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 0 --history_max 50 --dataset Beauty --encoder GRU4Rec --stage 2 --checkpoint blabla
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 0 --history_max 50 --dataset Beauty --encoder GRU4Rec --stage 0 --gamma 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 0 --history_max 50 --dataset Beauty --encoder GRU4Rec --stage 1 --temp 0.2 --batch_size 2048 --epoch 50 --early_stop 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 0 --history_max 50 --dataset Beauty --encoder GRU4Rec --stage 2 --checkpoint ../model/ContrastRec/Pre__Beauty__2019__encoder=GRU4Rec__temp=0.2__bsz=2048.pt
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 0 --history_max 50 --dataset Beauty --encoder GRU4Rec --stage 0 --gamma 1 --temp 0.2 --batch_size 4096
#
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Beauty --encoder Caser --stage 2 --checkpoint blabla
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Beauty --encoder Caser --stage 0 --gamma 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Beauty --encoder Caser --stage 1 --temp 0.2 --batch_size 2048 --epoch 50 --early_stop 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Beauty --encoder Caser --stage 2 --checkpoint ../model/ContrastRec/Pre__Beauty__2019__encoder=Caser__temp=0.2__bsz=2048.pt
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Beauty --encoder Caser --stage 0 --gamma 1 --temp 0.2 --batch_size 2048

# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Beauty --encoder BERT4Rec --stage 2 --checkpoint blabla
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Beauty --encoder BERT4Rec --stage 0 --gamma 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Beauty --encoder BERT4Rec --stage 1 --temp 0.2 --batch_size 2048 --epoch 50 --early_stop 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Beauty --encoder BERT4Rec --stage 2 --checkpoint ../model/ContrastRec/Pre__Beauty__2019__encoder=BERT4Rec__temp=0.2__bsz=2048.pt
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Beauty --encoder BERT4Rec --stage 0 --gamma 1 --temp 0.2 --batch_size 4096


# VideoGames
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Video_Games --encoder GRU4Rec --stage 2 --checkpoint blabla
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Video_Games --encoder GRU4Rec --stage 0 --gamma 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Video_Games --encoder GRU4Rec --stage 1 --temp 0.2 --batch_size 2048 --epoch 50 --early_stop 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Video_Games --encoder GRU4Rec --stage 2 --checkpoint ../model/ContrastRec/Pre__Video_Games__2019__encoder=GRU4Rec__temp=0.2__bsz=2048.pt
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Video_Games --encoder GRU4Rec --stage 0 --gamma 1 --temp 0.2 --batch_size 4096
#
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-4 --history_max 50 --dataset Video_Games --encoder Caser --stage 2 --checkpoint blabla
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-4 --history_max 50 --dataset Video_Games --encoder Caser --stage 0 --gamma 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Video_Games --encoder Caser --stage 1 --temp 0.2 --batch_size 512 --epoch 50 --early_stop 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Video_Games --encoder Caser --stage 2 --checkpoint ../model/ContrastRec/Pre__Video_Games__2019__encoder=Caser__temp=0.2__bsz=512.pt
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-4 --history_max 50 --dataset Video_Games --encoder Caser --stage 0 --gamma 1 --temp 0.2 --batch_size 1024

# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-4 --history_max 50 --dataset Video_Games --encoder BERT4Rec --stage 2 --checkpoint blabla
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-4 --history_max 50 --dataset Video_Games --encoder BERT4Rec --stage 0 --gamma 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Video_Games --encoder BERT4Rec --stage 1 --temp 0.2 --batch_size 512 --epoch 50 --early_stop 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Video_Games --encoder BERT4Rec --stage 2 --checkpoint ../model/ContrastRec/Pre__Video_Games__2019__encoder=BERT4Rec__temp=0.2__bsz=512.pt
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-4 --history_max 50 --dataset Video_Games --encoder BERT4Rec --stage 0 --gamma 1 --temp 0.2 --batch_size 2048


# Yelp-2018
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Yelp --encoder GRU4Rec --stage 2 --checkpoint blabla
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Yelp --encoder GRU4Rec --stage 0 --gamma 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Yelp --encoder GRU4Rec --stage 1 --temp 0.2 --batch_size 2048 --epoch 50 --early_stop 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Yelp --encoder GRU4Rec --stage 2 --checkpoint ../model/ContrastRec/Pre__Yelp__2019__encoder=GRU4Rec__temp=0.2__bsz=2048.pt
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Yelp --encoder GRU4Rec --stage 0 --gamma 5 --temp 0.2 --batch_size 4096
#
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Yelp --encoder Caser --stage 2 --checkpoint blabla
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Yelp --encoder Caser --stage 0 --gamma 0 --batch_size 512
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Yelp --encoder Caser --stage 1 --temp 0.2 --batch_size 2048 --epoch 50 --early_stop 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Yelp --encoder Caser --stage 2 --checkpoint ../model/ContrastRec/Pre__Yelp__2019__encoder=Caser__temp=0.2__bsz=2048.pt
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 1e-6 --history_max 50 --dataset Yelp --encoder Caser --stage 0 --gamma 0.1 --temp 0.2 --batch_size 512

# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-3 --l2 0 --history_max 50 --dataset Yelp --encoder BERT4Rec --stage 2 --checkpoint blabla
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Yelp --encoder BERT4Rec --stage 0 --gamma 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Yelp --encoder BERT4Rec --stage 1 --temp 0.2 --batch_size 2048 --epoch 50 --early_stop 0
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Yelp --encoder BERT4Rec --stage 2 --checkpoint ../model/ContrastRec/Pre__Yelp__2019__encoder=BERT4Rec__temp=0.2__bsz=2048.pt
# python main.py --model_name ContrastRec --emb_size 64 --lr 1e-4 --l2 1e-6 --history_max 50 --dataset Yelp --encoder BERT4Rec --stage 0 --gamma 5 --temp 0.2 --batch_size 4096
