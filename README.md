# Research projects on GANs for Data Augemtation of face texture maps images #

The followin repository containes the work reference in the blog post:

https://drive.google.com/open?id=1UU7B8_BHsDDTweno9AhwY_qdyGuLA6Up

The two main sub-projects are:

- 3d-model-fitting-pipeline: contains all the functions required to generate the texture map, and 3D mesh output of applying the a 3DFMM, which created the training dataset of the networks contained in the following folder.
- texture-map-synthesis: adapted versions of the following models to perform texture map completion: AGE, pix2ix/CycleGAN and pix2pixHD (pairPix2Pix and twinPix2Pix)

Additional folders include:

- utils - general utils used through the notebooks and fitting pipeline
- video-processing - notebooks with functions for the pipeline generating texture maps out of videos
- pytorch-cnn-visualizations - cloned library with several utils for visualizaing filters and (class) activation maps (CAMs) + notebooks visualizing the CAMs of DenseNet and pix2pixHD Discriminator
- dockerfiles - dockerfiles to recreate the environment for cuda 8 and cuda 9

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

For replicating the environment use the corresponding image in the dockerfiles folders (cuda 8.0 or cuda 9.0) in ```resources/dockerfiles```.

