# Subsampling_aliasing_artifact_eliminator
Artificial Intelligence-based Magnetic Resonance Image Reconstruction method using Generative adversarial network

## Abstract
Our method is a parallel MR imaging method using deep learning. Various studies are under way to reduce the scan time of MRI. Undersampling without getting all of the phase encoding lines can save a lot of scan time, but aliasing artifacts occur. We attempted to remove this artifact using neural networks. Although it succeeded in getting some gain with multilayer perceptron, we tried to go further using Generative Adversarial Networks.

## Dataset
For more detailed information and download, please refer to http://xai.kaist.ac.kr/research/data/

## Prerequisites
+ Python 3.6.6
+ Pytorch 0.4.1
+ Numpy 1.15.2
+ h5py, matplotlib


## Reference
```
@{
    author    = {Kinam Kwon, Dongchan Kim, and HyunWook Park},
    title     = {A parallel mr imaging method using multilayer perceptron},
    journal   = {Medical Physics},
    year      = {2017}
}
```


# XAI Project 

**This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2017-0-01779, A machine learning and statistical inference framework for explainable artificial intelligence)**

+ Project Name : A machine learning and statistical inference framework for explainable artificial intelligence(의사결정 이유를 설명할 수 있는 인간 수준의 학습·추론 프레임워크 개발)

+ Managed by Ministry of Science and ICT/XAIC <img align="right" src="http://xai.unist.ac.kr/static/img/logos/XAIC_logo.png" width=300px>

+ Participated Affiliation : UNIST, Korea Univ., Yonsei Univ., KAIST, AItrics  

+ Web Site : <http://openXai.org>

## Contacts
If you have any question, please contact Namho Jeong (nhjeong@kaist.ac.kr).
