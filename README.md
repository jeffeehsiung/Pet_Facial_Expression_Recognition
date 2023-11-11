# Trello

Link to trello is [here](https://trello.com/invite/b/PDlvtlED/ATTI80751ddf1d8b0471659f54c7dce4e5f7BDA5DE8F/funnymodel)

# What needs to change?

- Change title : Joerie ✅ (done)
- Change motivation : Jeffee/Joerie ✅ (done)
- Problem we tackling : Joerie ✅ (done)
- Methods : Jeffee/Joerie (Joerie part done)
- Solution : Joerie ✅ (done)
- Papers : Jeffee
- Update images and videos : Joerie
- Restructure readme : Jeffee

# What do we need to have unti next milestone?

We need to:
- Go through [This part](#💻-project-source-code) first repo and see if we can use it for our project
- Go through the lectures for nural networks
- What steps are we going to use?
- Additionally, YouTube tutorials are very useful
  - share the link if you find some interesting
- Start coding
  - Make plan:
    - who does what
    - who does paper
    - who does code
    - who does poster
- SHOW REUSE OF CODE FROM LABS
- Get theory why use this blablalba 2 + 2 = 4

# 🔍Bridging the Gap: Pet Facial Expression Recognition for Enhanced XR Human-Pet Interactions

<!--TOC-->

* [🚀 Motivation and Explanation of Title](#🚀-motivation-and-explanation-of-title)
    * [🤔 What problem are we tackling](#🤔-what-problem-are-we-tackling)
        * [🧮 Methods](#🧮-methods)
        * [💡 Solution](#💡-solution)
    * [🧐 Explanation of title](#🧐-explanation-of-title)
* [📚 Resources](#📚-resources)
    * [📑 Papers](#📑-papers)
    * [📊 Datasets](#📊-datasets)
    * [💻 Project Source Code](#💻-project-source-code)


## 🤔Motivation: what problem are we tackling

In extended reality (XR), the integration of pets into virtual spaces is has created for a wide range of new possiblities. However, there's still a significant void in understanding the emotianal nuances of pets in XR. We will try to enhance the technology by developing a model that (a) identifies between two different pet types (cats and dogs) and (b) classify between different facial expressions (happy, sad, and angry).

Motivated by the growth of XR based technologies and the value of pets in human mental health, we will aim to integrate pets into XR. This will not only accommodate the preferences of the user in question but also enhances the overall immersive experience in human-pet interaction.

**Potential use case**

Our model could find it's application in remote pet monitoring for healthcare assessment. For instance: by using 
<a href="https://en.wikipedia.org/wiki/Remote_camera">trail cameras</a> biologists, researchers, or even hobbyist could remotely look at the emotial state of an animal; conclude distress; behavioural change and so forth. We think this model has huge potential, especially when this model could get extended to multiple animals and multiple emotions.  

## 💡 Solution 

**Pre-trained model**: Using a pre-trained CNN (e.g. ResNet for example Joerie uses this in his thesis as well) can be used as a base. These models have shown their success in image classifcation to capture feature vectors.

**Custom model**: Using ML techinques/principles from the course we can make a customized CNN architecture with added convulutioonal layers and pool layers for feature extraction and spactial recudtion respectivily

## 🧮 Method

Our solution will have the following cronological steps


1. Load data by storing each image path (e.g. "```list[x] = "/images/happy/dog15.png```") in a list and it's corresponding label in another (```list2[x] = "happy"```)
2. Transform the lists into a dataframe
3. Exokiratiry Data Analysis (EDA) and analyze data to get more insights (just like we did in all the labs, explore data get familiar with it)
4. Make train, test, and validate sets
5. Make Data Generator (DG) for Train, Test and Validate set. We can use Tensorflow Generator for it.
6. Load the pre trained model; add some layers; compile. (here will our own code be insjected probaly)
7. Evaluate the result by plotting results (also like in the labs)

## First result

blablabla


# 📚 Resources

## 📑 Papers

<a name="paper1"></a>
[1] *Wood, E. et al. (2022). **3D Face Reconstruction with Dense Landmarks**. In: Avidan, S., Brostow, G.*, Cissé, M., Farinella, G.M., Hassner, T. (eds) Computer Vision – ECCV 2022. ECCV 2022. Lecture Notes in Computer Science, vol 13673. Springer, Cham.. 
[![DOI:10.1007/978-3-031-19778-9_10](https://zenodo.org/badge/DOI/10.1007/978-3-031-19778-9_10.svg)](https://doi.org/10.1007/978-3-031-19778-9_10)

<a name="paper2"></a>
[2] *Lawin, F. J., Moeller, M.-M., & Petersson, L*. (2017). **MVSNet: Depth Inference for Unstructured Multi-view Stereo.** Computer Vision and Pattern Recognition (CVPR).
[![arXiv](https://img.shields.io/badge/arXiv-1703.06870-b31b1b.svg)](https://arxiv.org/abs/1703.06870)

<a name="paper3"></a>
[3] *Fanzi Wu, Linchao Bao, Yajing Chen, Yonggen Ling, Yibing Song, Songnan Li, King Ngi Ngan, Wei Liu*. (2019). **MVF-Net: Multi-View 3D Face Morphable Model Regression**. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 959-968
[![arXiv](https://img.shields.io/badge/arXiv-1703.06870-b31b1b.svg)](https://arxiv.org/abs/1703.06870)

<a name="paper4"></a>
[4] *Vasquez-Gomez, J.I., Troncoso, D., Becerra, I. et al*. **Next-best-view regression using a 3D convolutional neural network**. Machine Vision and Applications 32, 42 (2021).
[![DOI:10.1007/s00138-020-01166-2](https://zenodo.org/badge/DOI/10.1007/s00138-020-01166-2.svg)](https://doi.org/10.1007/s00138-020-01166-2)

<a name="paper5"></a>
[5] *Y. Choi, Y. Uh, J. Yoo and J. -W. Ha*, **StarGAN v2: Diverse Image Synthesis for Multiple Domains**. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, 2020, pp. 8185-8194.
[![DOI:10.1109/CVPR42600.2020.00821](https://zenodo.org/badge/DOI/10.1109/CVPR42600.2020.008212.svg)](https://doi.org/10.1109/CVPR42600.2020.00821)

<a name="paper6"></a>
[6] *A. S. Jackson, A. Bulat, V. Argyriou and G. Tzimiropoulos*, **Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression**, 2017 IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 2017, pp. 1031-1039, doi: 10.1109/ICCV.2017.117.
[![DOI:10.1109/ICCV.2017.117](https://zenodo.org/badge/DOI/10.1109/ICCV.2017.117.svg)](https://doi.org/10.1109/ICCV.2017.117)

<a name="paper7"></a>
[7] *J. Zhang, H. Hu and S. Feng*, **Robust Facial Landmark Detection via Heatmap-Offset Regression**, in IEEE Transactions on Image Processing, vol. 29, pp. 5050-5064, 2020.
[![DOI:10.1109/TIP.2020.2976765](https://zenodo.org/badge/DOI/10.1109/TIP.2020.2976765.svg)](https://doi.org/10.1109/TIP.2020.2976765)

## 📊 Datasets
<a name="pet's facial expression"></a>
* [Kaggle 🐶Pet's Facial Expression Image Dataset😸](https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset/data) 
* [Kaggle Animal faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces/data) 

## 💻 Project Inspiration
* [CNN | Beginners | 🐶Pet's Expression Recognition](https://www.kaggle.com/code/anshtanwar/cnn-beginners-pet-s-expression-recognition) 
* [500 ML Project](https://github.com/ashishpatel26/500-AI-Machine-learning-Deep-learning-Computer-vision-NLP-Projects-with-code)
<a name="facial_landmark"></a>
* [Facial Landmark and Image Morhphine: Species](https://github.com/emreslyn/facial_landmark_and_image_morphing)
