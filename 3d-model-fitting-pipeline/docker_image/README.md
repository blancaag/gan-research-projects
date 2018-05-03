# README #

Dockerfile for building an image with the following specifications:

* Ubuntu 16.04
* cuda:8.0
* cudnn6
* gcc-17, cmake 3.10
* Python bindings for the project patrickhuber/eos project [see: https://github.com/patrikhuber/eos]
* Miniconda3
* Others: Pytorch (for cuda:8.0), OpenCV, dlib.

### How do I get set up? ###

* Requirements: Docker
* docker build . -f Dockerfile -t nvidia_mini
* basic: docker run -it -p 8888:8888 nvidia_mini
* mounting a dir: docker run -it -v ~/project:/home/project --user root -p 8888:8888 nvidia_mini
