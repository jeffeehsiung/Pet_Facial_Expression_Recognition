# 🔍Efficient Multi-View 3D Model Reconstruction Using CNN Regression

<!--TOC-->

* [🚀 Motivation and Explanation of Title](#motivation-and-explanation-of-title)
    * [🤔 What problem are we tackling](#what-problem-are-we-tackling)
        * [💡 Solution](#solution)
    * [🧐 Explanation of title](#explanation-of-title)
* [📚 Resources](#resources)
    * [📑 Papers](#papers)
    * [📊 Datasets](#datasets)
    * [💻 Project Source Code](#project-source-code)


## 🚀 Motivation and explenation of title

First of all the motivation behind our project is rooted in the love for animals and the intrest in expanding our knowledge of field of computer vision. When browdsing the web for ideas we came across human reconstruction, and thought why not with animals.


### 🤔 What problem are we tackling

Traditional 3D reconstruction methods ofter require expensive and time-consuming technology, like 3D-scanning <a href="https://en.wikipedia.org/wiki/3D_scanning">Source to Wikipedia</a>, or even manual reconstriuction in 3D software (Inventor, AutoCad, Solidworks).
Which are not feasible for large-scale applications or capturing animals in their natural habitats.


#### 💡 Solution

We will try to use the power of ML to make this possible.

By doing so, researchers, biologist, vetnarie, and educatorscan easily access 3D images/models of animals for various purposes: anatomy, sick-ness behaviour, behavior in general, or even creating realistic simulations.

Even more so we think this has a potential to be using the gaming industry, where 3D models are used to create realistic simulations of animals. Think about the MetaVerse, you want your pet to be in the game?  You don't want to spend hours creating it? Just take a few pictures of it and the game will create a 3D model of it.


### 🧐 Explanation of title

 - **Efficient Multi-View**: The model is able to reconstruct the animal from multiple images. This is a very important feature since we want to be able to reconstruct the animal from multiple images.
 - **3D**: The model is able to reconstruct the animal in 3D. This is a very important feature since we want to be able to reconstruct the animal in 3D. (This will be in a simple sparce 3D face modeling format) see picture from paper below.
 - **Reconstruction**: Talks for itself

- **CNN Regression**: The model uses a CNN to regress the 3D model from the input images. This is a very important feature since we want to be able to reconstruct the animal from multiple images<a href="#paper2"> [2]</a><a href="#paper3"> [3]</a><a href="#paper4"> [4]</a>.

<p align="center">
    <img src="sparse_3D_recon.jpeg" alt="3D face model">
    <br>
    From paper <a href="#paper1"> [1]</a>
</p>

# 📚 Resources

## 📑 Papers

<a name="paper1"></a>
[1] (Wood, E. et al. (2022). 3D Face Reconstruction with Dense Landmarks. In: Avidan, S., Brostow, G., Cissé, M., Farinella, G.M., Hassner, T. (eds) Computer Vision – ECCV 2022. ECCV 2022. Lecture Notes in Computer Science, vol 13673. Springer, Cham. https://doi.org/10.1007/978-3-031-19778-9_10)

<a name="paper2"></a>
[2] (Lawin, F. J., Moeller, M.-M., & Petersson, L. (2017). MVSNet: Depth Inference for Unstructured Multi-view Stereo. Computer Vision and Pattern Recognition (CVPR). https://arxiv.org/abs/1703.06870)

<a name="paper3"></a>
[3] (Fanzi Wu, Linchao Bao, Yajing Chen, Yonggen Ling, Yibing Song, Songnan Li, King Ngi Ngan, Wei Liu; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 959-968)

<a name="paper4"></a>
[4] (Vasquez-Gomez, J.I., Troncoso, D., Becerra, I. et al. Next-best-view regression using a 3D convolutional neural network. Machine Vision and Applications 32, 42 (2021). https://doi.org/10.1007/s00138-020-01166-2)


## 📊 Datasets
* [Kaggle face recognition dataset collection](https://www.kaggle.com/datasets?search=fac&tags=13207-Computer+Vision) 
* [Kaggle Facial keypoint detection](https://www.kaggle.com/datasets/nagasai524/facial-keypoint-detection) 
* [Kaggle Animal faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces/data) 

## 💻 Project Source Code
* [3D Face Reconstruction using CNN](https://github.com/AaronJackson/vrn) 
* [500 ML Project](https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code)
