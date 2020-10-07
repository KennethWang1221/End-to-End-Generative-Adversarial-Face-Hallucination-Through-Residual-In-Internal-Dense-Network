# End-to-End Generative Adversarial Face Hallucination Through Residual in Internel Dense Network

### Abstract

Face Hallucination has been a highly attractive computer vision research in recent years. It is still a particularly challenging task since human face has complex and sensitive structure. In this paper, we proposed a novel en d-to-end Generative Adversarial Face Hallucination Residual in Internel Dense Network (FH-RIDN) to hallucinate unaligned very Low-resolution(LR) face images (32x32x3) to its 8x High-Resolution(HR) counterparts (256x256x3). Our proposed method, FH-RaGAN, consists of generator with novel arthitecture and improved discriminator. Generator is supposed to generate visually pleasant hallucinated face images, while discriminator aims to evaluate how much input images are realistic. With constant adversarial learning, FH-RaGAN is able to hallucinate perceptually plausible face images. Extensive experiments on large face datset demonstrate that the proposed method significantly outperforms the state-of-the-art.



## Introduciton

Face Hallucination (FH), also know as, Face Super Resolution(FSR),  is a domain-specific image SR problem, which refers to hallucinate High-Resolution (HR) face image from its Low-Resolution (LR) counterpart. FH is an significant task in face analysis field , which is of remarkable benefit to computer vision applications such as face surveillance and recognition. However, Face Hallucination is an ill-posed inverse problem and praticularly challenging since LR images may correspond to many HR candidate images and LR images have lost many crucial facial structures and components. In order to hallucinate high quality face images, many FH methods have proposed. Generally, we classify FH approachs into two categories: traditional based and deep learning based FH method.

Many traditional studies have been conducted to address FH tasks. Based on face image samples learning, Baker and Kanade[1] learn the best relationship between LR and HR pathes to reconstruct high-frequency details of LR face images. In[2], wang and Tang employ eigen-transformation to build linear mapping between HR and HR face subspaces. Liu et al. [3] learn a global model based on PCA for face hallucination to improve the facial details. By adopting relationship between particular facial components, Yang et al. [4] combine the face priors to recover facial information from HR image components. 

Recently, deep learning based super resolution (SR) methods have emerged and achieved the state-of-the-art performance [5,6,7]. Dong et al[5]. firstly introduce deep learning based super resolution method named SRCNN that directly learns an end-to-end mapping between HR images and LR images by three convolutional layers. The novel Bi-channel convolutional network is presented by Zhou et al.[6] to hallucinate face images in the wild. Zhu et al.[7] presented the Cascaded Bi-Network(CBN). In CBN, two sub-networks (face hallucination and dense correspondence filed estimation) are optimized alternatively.

The limition of above SR methods is that they utilize reconstruction loss such as $$L1$$ or $$L2$$ to optimize hallucination process, which lead to produce over-smooth hallucinated images even though these models obtain higher Peak Signal-to-Noise Ratio (PSNR) value.  To address severe problem, several Generative Adversarial Network (GAN) - based SR models are proposed [8, 9, 10, 11, 13, 14]. It is approval that GAN-based SR models using adversarial loss are capable of further generating visually realistic HR images. Christian et al.'s work first extend GAN to the field of SISR and proposed an effective method, called SRGAN using perceptual loss and adversarial loss. Following by SRGAN, Wang et al. presents enhanced version named ESRGAN by proposing new generator architecture and using improved perceptual loss. Yu and Porikli [x] employs generator combined deconvolution operations to ultra-resolve a tiny face images, which can be called URDGN. 

However, The aforementioned GAN-based FH models are prone to model collapse, resulting in artifacts in the hallucinated results, especially once the input image resolution is extremely low, the hallucinated images lose fine details and become non-realistic. To address this problem, in this paper, we prpose a novel GAN-based FH method, Face Hallucination based on Relativistic Average Generative Adversarial Network (FH-RaGAN).  

In summary, the contributions of this paper are mainly in five aspects:

1. Our proposed method is capable of hallucinating a unaligned tiny (i.e. 32x32x3) LR face image to hallucinated version (256x256x3) with an ultra upscaling factor 8x.

2. we present a Residual in Internel Dense Block, which boosts the flow of feature through the generator and provide abundant feature for the hallucination process.

3. we exploit relativistic average discriminator (RaD), which evaluates the probability that the given HR images is more realistic than super-resolved images. 

4. Contrary to the classical FH methods, our method does not involve any prior information or claim facial landmark points for its hallucinating, which facilitate the whole training process and add model robustness

5. we compare with state-of-the-art methods, and demonstrate that our method outerforms these approaches regarding both qualitatively and quantitatively.

## Proposed Method ##

In this section, our proposed method FH-RaGAN will be presented in detail. FH-RaGAN  mainly consists of two parts: novel network architecture and improved loss function. We present FH-RaGAN architecutre first and then discuss the improvement of the discriminator and perceptual loss.

Where the goal of  $D$ is to evaluate how much input images ($$I_{HR}$$ / $$I_{SR}$$ ) are realistic. while $$G$$ is supposed to produce hallucinated images $$I_{SR}$$ that similar to $$I_{HR}$$

### Architecture of GAFH-RIDN ###

![ICASSP_Overview (1)](/Users/wangxiang/Downloads/ICASSP_Overview (1).png)

Figure 1.  The architecture of our Face Hallucination Relativistic Average Generative Adversarial Network (FH-RaGAN). $I_{HR}$ and $I_{FH}$ denote HR face images and face hallucinated images respectively. DNB describes Dense Nested Block, as shown in Fig. 2; K, n, s represent kernel size, number of feature maps and strides correspondingly.

#### Generator

As shown in the top of Fig 1. , the proposed generator mainly consists of the three stages: shallow feature extraction (SFE), multi-level dense bock module (MDBM) and upsampling module (UM). The LR face image $I_{LR}$ is passed to  shallow feature layer initially and obtains Hallucinated face image $I_{HF}$ from upsampling module. Regarding to the shallow feautre extraction, we utilize one convolution layer with kernel size 9x9 to extract shallow feature maps from $I_{LR}$. It can be expressed as follows:

$$F_{SFE}=f_{SFE}(I_{LR})$$

where $f_{SFE}(\cdot)$ represents feature extraction in SFE.  $F_{SFE}$ denotes the shallow (low-level) feature and serve as the input to the MDBM. The follow module MDBM is built up by multiple Dense Nested Block (DNB) formed by Residual in Internel Dense Block(RIDB), as shown in Fig. 2, which will be discussed in next subsection. This procedure is formulated as :

$$F_{DNB\_i}=f_{DNB}(F_{DNB\_i-1}),i=1,2,...,n$$ 

where $f_{DNB}$ denotes high-level feature extraction in DNB, $F_{DNB\_i-1}, F_{DNB\_i}$ denotes the input and output of  $i$-th DNB respectively. We fusion low-level featuers and high-level features to boost hallucination performance via dense skip connection. This process can be expressed as:

$$F_{fusion}=C_{conv}(F_{DNB\_n})+F_{SFE}$$ 

where $C_{conv}$ denotes a convolution operation. Futhermore, the fusion feature $F_{fusion}$ is transformed from the LR sapce to HR sapce through upsampling layers in UM. The reconstruction process can be formulated as :

$$I_{HF}=f_{UM}(F_{fusion})$$

where $f_{UM}$ represents the upsampling operation. Finally, we obtain hallucinated face image $I_{HF}$.

##### 3.2 Residual in Internel Dense Block/in order to obtain abundant feature information, 

As described in Sec.1, we proposed a novel unit RIDB in generator. The architecture of RIDB is inspired by[SRGAN,ESRGAN], while a commonly encountered issue in these models is that the vanishing gradient problem occurs, resulting in the generator produces ghosting artifacts. To address this problem we make modification to further improve the reconstruction quality. By involving residual learning and many dense connections in RIDB, the flow of feature in generator can be strengthen. RIDB is made up by four internel blocks, where each internel block is composition of two groups of a convolution layer followed by LeakyReLU activation layer. Owing to the dense skip connections, the feature maps of each layer are propagated into all succeeding layers, promoting an effective way to fusion low-level features and high-level features. The architecture of the RIDB can be expressed as:

$$F_{i,j} =\delta(W_{i,j}[F_{i-1},F_{i,1},...,F_{i,j-1}])$$

Where $F_{i,j}(\cdot)$ represents the output of $i$-th Conv layer of $j$-th RIDB. $F_{i-1},F_{i}$ denotes the input and output of the preceding $i$-th RIDB respectively. $[F_{i-1},F_{i,1},...,F_{i,j-1}]$ refers to the concatenation of feature maps. $W_{i,j}$ is the weights of the $i$-th Conv layer. $\delta$ Denotes the LeakyReLU [x] activation layer.  With the help of RIDB, our proposed method is capable of obtaining abundant feature information and alleviating the vanishing-gradient problem.



![ICASSP_RIDB](/Users/wangxiang/Downloads/ICASSP_RIDB.png)

​	**Figure 2. ** **Top**: Dense Nested Block(DNB) stacked by multiple RIDB. **Bottom**: the architecture of our proposed Residual in Internel Dense Block (RIDB).



#### Improvement Discriminator

Inspired by [RaGAN], We adopt the Relativistic average discriminator (RaD) in our method instead of the discriminator of standard GAN (SGAN) \cite{GAN}, SRGAN \cite{SRGAN} and URDGN \cite{URDGN}. Owing to RaD, the discriminator of GFH-RIDN has the ability to distinguish how the given HR image is more authentic compared with the hallucinated image. The architecture of discriminator is shown in the bottom section of Fig. 1. The restriction of a discriminator in \cite{GAN,SRGAN,URDGN} is that they only concentrate on increasing the probability that fake samples belong to real rather than decreasing the probability that real samples belong to real simultaneously, which omits the prior knowledge that half bach images are real and the other half are fake samples. In other words, if the discriminator ignores real samples during the learning procedure, as a result, the model can not provide sufficient gradient when updating the generator, which encounters the problem of gradient vanishing for training generator in the learning process. The SGAN discriminator can be expressed as: 

$$D(x) = \sigma (C(x))$$ (1) 

As Equation. 1 shown, standard discriminator only evulates the probability for a given real sample or a generated sample. Where $x$ can be either $I_{HR}$ or $I_{SR}$ in this context; $\sigma$ represents the sigmoid function which translates the results into a probability between 0 to 1. $C(x)$ denotes the probability predicted by  standard discriminator. From [x], it shows that  relativistic discriminator takes into consideration that how a given real sample is more authentic compared with a given generated sample. The relativistic discriminator is formulated as:

$$D(x_{r},x_{f}) = \sigma (C(x_{r}) - E_{x_{f}}[C(x_{f})])$$ (2)

Contrary to SGAN, the probability predicted by RaD relys on both $x_{r}$ and $x_{f}$ , which is capable of making discriminator relativistic.  In our method, a generator of FH-RaGAN is able to beneficial from gradients of both $I_{HR}$ and $I_{SR}$ in the adversarial training, therefore it facilitates generator recover clearer edges and texture.   Where $E_{x_{f}}$ denotes the average of the fake samples in one batch predicted by RaD. Accoring to Equation 5, We can optimize discriminator by adversarial loss $L_{D}^{adv}$. A generator is updated by $L_{G}^{adv}$ , as in Equation 6.

$$ L_{D}^{adv}= -\mathbb{E}_{I_{HR}}\sim  p_{(I_{HR})}\left [ log\left ( D(I_{HR},I_{SR}\right ))\right ] -\mathbb{E}_{I_{SR}}\sim p_{(I_{SR})}\left [ log\left( 1-D\left ( I_{SR},I_{HR} \right )\right )\right ]$$ 							(5)



$$ L_{G}^{adv}= -\mathbb{E}_{I_{HR}}\sim p_{(I_{HR})}\left [ log\left (1- D(I_{HR},I_{SR}\right ))\right ]-\mathbb{E}_{I_{SR}}\sim p_{(I_{SR})}\left [ log\left(D\left ( I_{SR},I_{HR} \right )\right )\right ]$$							 (6)

Owing to this property, our prposed method is capable of allowing the probability of $I_{HR}$ being real decrease while letting the probability of $I_{SR}$ being real increase, as a result FH-RaGAN is remarkable more stable and generate higher quality face images. Where $I_{HR}$ and $I_{SR}$ denotes HR images and hallucinated images respectively; $D(\cdot)$ describes the probability predicted by discriminator; $\mathbb{E}$ represents the exception; $I_{HR} \sim P_{I_{HR}}$$ and $ $I_{SR} \sim P_{I_{SR}}$ represents the HR images distribution and hallucianted images distribution respectively.

Our discriminator contains 9 Conv layers with the number of 3x3 filter kernels and the stride 1 or 2 alternately. The channels of feature maps increasing by a factor of 2, from 64 to 512. The resulting 512 feature maps are passed through two dense layers. Finally, after the sigmoid activation layer, RaD predicts the probability that whether the HF image is real or fake. 

##### Content Loss

Taking advantage of content loss $L_{content}$ is able to further promote detail enhancement . We adopt the pre-trained VGG-19 as feature extractor to obtain feature representation which used to calculate  $L_{content}$. We extract low-level feature maps of HR, FH images respectively obtained by the $3^{th}$ convolution layer before the $5^{th}$ maxpooling layer. HR and FH feature maps are defined as $\phi_{3,4}$ . $L_{content}$ is defined as follows:

$$L_{content} = \frac{1}{WH}\sum_{h=1}^{H}\sum_{w=1}^{W}\left \|\phi_{3,4} (I_{HR})-\phi_{3,4} (G(I_{LR})) \right \|^{2}$$																		  (4)

Where $\phi(\cdot)$ represents the feature extractor and $C$ denotes the number of channels and $W,H$ denotes the dimensions of  the feature maps. 

#### Loss Function

The total loss function $L_{perceptual}$ can be represented as two parts:   $L_{content}$ and $L_{adversarial}$. For the propose of improving the fidelity of the face images, we introduce the $L_{content}$ . In addition, an adversarial loss $L_{adversarial}$ is expected to enhance perceptual quality from visual aspect. The formulas are as follows, Content loss: sharp visually appealing results

$$ L_{perceptual} = \alpha L_{content} + \beta L_{G}^{adv}$$																															 (3)

Where $\alpha$ and $\beta$ were corresponding fixed parameters used to balance $L_{content}$ and $L_{adversarial}$. We set $\alpha$ = 1 ,$\beta$ = $10^{-3}$ respectively.

##### 

## Experiments

#### Datasets

We conducted experiments on public large-scale face image datasets, CelebA, most of which are frontal facial images. It consists of 202,599 face images of 10,177 celebrities. In our experiments, 162,048 HR face images are randomly selected as the training set, and the next 40,511 images were used as the testing set. 

#### Implementing Details

We completed two groups of experiments to verify the effects of our method when the upscaling factor is 4x and 8x. In order to obtain two groups of LR downsampled face images, we used bicubic interpolation kernel with downsampling factor $r$ =4 to LR 64x64x3,and factor $r$=8 to LR 32x32x3.  We first normalized the HR and LR images which are input of generator to convert the pixel values to a range between[-1,1]. We involved $tanh$ activation layer at the end of the generator, while tanh activation layer squashes input values to the same range. As a result of facilitating the calculation of the loss, since it was necessary to have all values in the same range. Thefore we converted the input image values to a range of between -1 to 1 before feded into generator. 

we implemented the proposed method with Keras using Tensorflow backend and trained on a NVIDIA 1080Ti GPU. In the training phase, We used Adam with $\beta_{1}$ = 0.9, $\beta_{2}$ = 0.999 to update the model . The learning rate is set to $10^{-4}$ . Finally, we trained our model for 10,0000 epochs. Taking into account the hardware conditions of our GPU, we set the batchsize to 8 as a maximum batch for training procedure. We alternately updated the generator and discriminator.

#### Comparisons with state-of-the-Art Methods

In the experiments, we taken the super-resolved face images when upsampling factor set to 4x, and 8x for comparison. PSNR(Peak Signal to Noise Ratio) and SSIM(Structural similarity)  played the role of  the quality evaluation metrics in our experimental. We adopt the state-of-the-art methods in the field of face super-resolution [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] to compare with our method quanlitatively and qualitatively. We involve the experimental results which are demonstrated in [8,15,16]. Noted that the evolution results of the state-of-the-Art methods when upsampling factor set to 4x in Table 1 are quoted from [8,15,16],  because they did not provided experiment results for scale factor 4x, and [4,5,8,] did not exploit SSIM as evaluation criterion, therefore the 4x super-resolved results of those methods in Table 1 are missing. For the qualitative comparison, as decipted in Fig 5 and Fig 6, we choose the traditional super-resolution method, bicubic, and the GAN based super-resolution method, SRGAN and ESRGAN. In order to compare the visual effects fairly, we used the published code of the above model, following the training and testing hyperparameters introduced in the paper, and used the same training set to train all models.

##### Quantitative Comparison

![截屏2020-10-07上午9.43.53](/Users/wangxiang/Desktop/截屏2020-10-07上午9.43.53.png)

**Table1. Quantitative comparison on CelebA dataset for scale factor x8 in terms of average PSNR(db) and SSIM. Numerical  in bold are indicated the best evaluation results among state-of-the-art methods.**

The quantitative comparison among state-of-the-art methods were demonstrated in Table 1. It shown the average PSNR and SSIM criterion of these methods on the CelebA dataset for scale factor 8x. As was shown in the Table 1, for the scale factor 8x, our method was the highest at 23.68db in terms of PSNR, followed by the second best[23] , FSRFCH  23.14db. For SSIM metrics, SEGAN was 0.02 higher than the second-place method, FSRFCH[23]. Compared with the classic face super-resolution method TDN[4], SEGAN was 1.02db and 0.04 higher in the values of PSNR and SSIM, respectively. It was apparent from the quantitative comparison that our method outperformed other state-of-the-art methods for scale factor and 8x in terms of PSNR and SSIM criterion.

##### Qualitative Comparison



![icassp_4x_result](/Users/wangxiang/Downloads/icassp_4x_result.png)

**Fig5. Qualitative comparison of 4x super-resolved face images on CelebA dataset. Column: (a) HR ground truth; (b) Input LR; (c) Bicubic (d) SRGAN (e) ESRGAN (f) Our method.** 



![icassp_8x_result](/Users/wangxiang/Downloads/icassp_8x_result.png)

**Fig6. Qualitative comparison of 8x super-resolved face images on CelebA dataset. Column: (a) HR ground truth; (b) Input LR; (c) Bicubic (d) SRGAN (e) ESRGAN (f) Our method.** 

Qualitative results were depicted in Fig5. and Fig6. It can be seen from Figure 5 that the LR images shown quite blurry visual characteristics. Through experimental figure, it was clear that the results of Bicubic interpolation contained over-smooth visualization effects which was unfaithful to its ground truth. Morever, compared with the Bicubic interpolation, reconstructed facial images of SRGAN were closer to HR images slightly, but they still failed to generate super-resolved images with fine details especially in eyes, and mouth region. It was obvious that ESRGAN outputs a generated images which was blurry and appears apparent artifacts. As was exhibited in Fig. 5 our method was able to super-resolve tiny images so that resulting SR facial images retained facial details and faithful to ground truth features effectively. From Figure. 6, it apparent presented the SR visual results for scale factor 8x. By comparison, the outputs of Bicubic interpolation failed to recover high frequency texture on account of limited by the adjacent pixels in the LR images. SRGAN encountered mode collapse problem during super-resolving process, so that it produced ghosting artifacts in SR images and unfaithful to the authentic face images. The reconstructed images generated by ESRGAN lacked high-frequency details and exhibited wrong shapes or random ghosting. Our method was capable of producing photo-realistic super-resolved  images which preserved perceptually sharper edges and  fine facial texture for both scale factor 4x, and 8x.

From quanlitative and qualitative comparison, it was fully proved that SEGAN was able to exploit global facial features of the reference images, and better restore the local facial components so that recovering high quality face images.

## Conclusion

In this paper, we propose a novel end-to-end manner trainable face hallucination method (GAFH-RIDN) to hallucinate extremely low-resolution(16x16 pixels) unaligned face images to its 8x high resolution version. By exploiting Residual in Internel Dense Block (RIDB) and Relativistic discriminator (RaD),  our method successfully  produces photo-realistic hallucinated face image. Extensive experiments demonstrate that GAFH-RIDN is superior to the state-of-the-art on benchmark face dataset qualitatively and qualitatively.

## References

1. J. Yang, L. Luo, J. Qian, Y. Tai, F. Zhang, and Y. Xu,“Nuclear  norm  based  matrix  regression  with  applica-tions  to  face  recognition  with  occlusion  and  illumina-tion changes,”IEEE Transactions on Pattern Analysisand Machine Intelligence, vol. 39, no. 1, pp. 156–171,2017.
2. Yu Chen, Ying Tai, Xiaoming Liu, Chunhua Shen, andJian  Yang,    “Fsrnet:   End-to-end  learning  face  super-resolution with facial priors,” 2017
3. X. Yu, B. Fernando, R. Hartley, and F. Porikli,  “Super-resolving very low-resolution face images with supple-mentary attributes,”  in2018 IEEE/CVF Conference onComputer  Vision  and  Pattern  Recognition,  2018,  pp.908–917
4. Xin Yu and Fatih Porikli,  “Face hallucination with tinyunaligned images by transformative discriminative neu-ral networks,” inThirty-First AAAI conference on artifi-cial intelligence, 2017.
5. Ashish Bora, Ajil Jalal, Eric Price, and Alexandros G.Dimakis,  “Compressed sensing using generative mod-els,” 2017
6. Xiaogang Wang and Xiaoou Tang,  “Hallucinating faceby  eigentransformation,”IEEE  Transactions  on  Sys-tems, Man, and Cybernetics, Part C (Applications andReviews), vol. 35, no. 3, pp. 425–434, 2005.
7. Ce Liu, Heung-Yeung Shum, and William T Freeman,“Face  hallucination:   Theory  and  practice,”Interna-tional Journal of Computer Vision,  vol. 75,  no. 1,  pp.115–134, 2007.
8.  Chih-Yuan  Yang,  Sifei  Liu,  and  Ming-Hsuan  Yang,“Structured  face  hallucination,”inProceedings  ofthe IEEE Conference on Computer Vision and PatternRecognition, 2013, pp. 1099–1106.
9.  C. Dong, C. C. Loy, K. He, and X. Tang, “Image super-resolution  using  deep  convolutional  networks,”IEEETransactions on Pattern Analysis and Machine Intelli-gence, vol. 38, no. 2, pp. 295–307, 2016.
10. Jiwon Kim, Jung Kwon Lee, and Kyoung Mu Lee, “Ac-curate image super-resolution using very deep convolu-tional networks,” 2016
11. Erjin Zhou, Haoqiang Fan, Zhimin Cao, Yuning Jiang,and Qi Yin,  “Learning face hallucination in the wild,”inProceedings of the Twenty-Ninth AAAI Conference onArtificial Intelligence, 2015, pp. 3871–3877
12. Shizhan Zhu, Sifei Liu, Chen Change Loy, and XiaoouTang,   “Deep  cascaded  bi-network  for  face  hallucina-tion,” 2016
13. Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero PSimoncelli,  “Image quality assessment: from error vis-ibility  to  structural  similarity,”IEEE  transactions  onimage processing, vol. 13, no. 4, pp. 600–612, 2004
14. Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Ca-ballero,  Andrew  Cunningham,  Alejandro  Acosta,  An-drew  Aitken,  Alykhan  Tejani,  Johannes  Totz,  ZehanWang,  and  Wenzhe  Shi,“Photo-realistic  single  im-age super-resolution using a generative adversarial net-work,” 2017.
15. Xintao  Wang,  Ke  Yu,  Shixiang  Wu,  Jinjin  Gu,  YihaoLiu, Chao Dong, Yu Qiao, and Chen Change Loy,  “Es-rgan:  Enhanced super-resolution generative adversarialnetworks,”  inProceedings of the European Conferenceon Computer Vision (ECCV), 2018, pp. 0–0
16. Xin Yu and Fatih Porikli,  “Ultra-resolving face imagesby discriminative generative networks,”   10 2016, vol.9909, pp. 318–333.
17. X. Yu and F. Porikli, “Hallucinating very low-resolutionunaligned and noisy face images by transformative dis-criminative  autoencoders,”   in2017  IEEE  Conferenceon Computer Vision and Pattern Recognition (CVPR),2017, pp. 5367–5375.
18. Richard  Zhang,   Phillip  Isola,   Alexei  A.  Efros,   EliShechtman, and Oliver Wang, “The unreasonable effec-tiveness of deep features as a perceptual metric,” 2018.
19. Alexia Jolicoeur-Martineau,  “The relativistic discrimi-nator: a key element missing from standard gan,” 2018.
20. Xiang Ma, Junping Zhang, and Chun Qi, “Hallucinatingface by position-patch,”Pattern Recognition,  vol. 43,no. 6, pp. 2224–2236, 2010.Xiang Ma, Junping Zhang, and Chun Qi, “Hallucinatingface by position-patch,”Pattern Recognition,  vol. 43,no. 6, pp. 2224–2236, 2010.
21. Tong Tong, Gen Li, Xiejie Liu, and Qinquan Gao,  “Im-age super-resolution using dense skip connections,”  inProceedings  of  the  IEEE  International  Conference  onComputer Vision (ICCV), Oct 2017.
22. Ian J. Goodfellow,  Jean Pouget-Abadie,  Mehdi Mirza,Bing  Xu,  David  Warde-Farley,  Sherjil  Ozair,  AaronCourville, and Yoshua Bengio,  “Generative adversarialnetworks,” 2014
23. Xin  Yu,   Basura  Fernando,   Bernard  Ghanem,   FatihPorikli,  and  Richard  Hartley,    “Face  super-resolutionguided  by  facial  component  heatmaps,”    inProceed-ings  of  the  European  Conference  on  Computer  Vision(ECCV), 2018, pp. 217–233.
24. Justin Johnson, Alexandre Alahi, and Li Fei-Fei,  “Per-ceptual  losses  for  real-time  style  transfer  and  super-resolution,” 2016
25. Alexey Dosovitskiy and Thomas Brox, “Generating im-ages  with  perceptual  similarity  metrics  based  on  deepnetworks,” 2016.
26. Karen Simonyan and Andrew Zisserman,   “Very deepconvolutional  networks  for  large-scale  image  recogni-tion,” 2015.
27. Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang,“Deep learning face attributes in the wild,” 2015.