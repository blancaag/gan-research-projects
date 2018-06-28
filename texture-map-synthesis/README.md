# README #

The following repository contains the work referenced in the blog post:

https://drive.google.com/open?id=1UU7B8_BHsDDTweno9AhwY_qdyGuLA6Up

This preprository is structured as follows:

- texture-map-synthesis: adapted versions of the following models to perform texture map completion for
	* AGE
	* pix2pix/CycleGAN
	* pix2pixHD 
	* pairPix2Pix
	* twinPix2Pix

### How to run the code? ###

Examples for running the training/test code of each project (please, refer to the options of each one):

* pix2pixHD (pairPix2Pix, twinPix2Pix)

```CUDA_VISIBLE_DEVICES=X python train.py --dataroot /workspace/blanca/training/ --dataset_list train val --loadSize 256 --fineSize 256 --nThreads 16 --display_port 9006 --name run_name```

```CUDA_VISIBLE_DEVICES=X python test.py --name run_name --dataroot /blanca/training_datasets/pix2pix --dataset_list test_datasets/labeled/bad-fits --loadSize 256 --fineSize 256 --label_nc 0 --no_instance --which_epoch 100```

* pix2pix/CycleGAN

```CUDA_VISIBLE_DEVICES=X python train.py --dataroot ../data --model pix2pix --dataset_mode aligned --pool_size 50 --lr=0.0002 --lambda_A=200 --name run_name```

```CUDA_VISIBLE_DEVICES=X python test.py --dataroot ../data --model pix2pix --name run_name --dataset_mode aligned --norm batch```

* AGE

```CUDA_VISIBLE_DEVICES=X python train.py```
```CUDA_VISIBLE_DEVICES=X python test.py```



