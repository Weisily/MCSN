#-------------MCSN train 
#python main.py --model MCSN--scale 2 --save mcsn_x2  --n_resblocks 3  --lr 1e-4  --n_feats 64 --res_scale 1 --batch_size 16 --n_threads 6 
#python main.py --model MCSN--scale 3 --save mcsn_x3  --n_resblocks 3  --lr 1e-4  --n_feats 64 --res_scale 1 --batch_size 16 --n_threads 6 
#python main.py --model MCSN--scale 4 --save mcsn_x4  --n_resblocks 3  --lr 1e-4  --n_feats 64 --res_scale 1 --batch_size 16 --n_threads 6 
#python main.py --model MCSN--scale 8 --save mcsn_x8  --n_resblocks 3  --lr 1e-4  --n_feats 64 --res_scale 1 --batch_size 16 --n_threads 6 
# Test benchmarkPlus
#-------------------------------------------------
#python main.py --model MCSN --data_test Set5+Set14+B100+Urban100+Manga109  --scale 4 
                           --pre_train ../experiment/mscn_x4/model/model_best.pt --test_only  --self_ensemble

# Test your own images
#--------------------------------------------------
#python main.py --dir_demo Demo --scale 4 --pre_train  ../experiment/mcsn_x4/model/model_best.pt --test_only --save_results 
