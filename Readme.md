# End-to-End Generative Adversarial Face Hallucination Through Residual in Internel Dense Network

### Abstract

Face Hallucination has been a highly attractive computer vision research in recent years. It is still a particularly challenging task since human face has complex and sensitive structure. In this paper, we proposed a novel en d-to-end Generative Adversarial Face Hallucination Residual in Internel Dense Network (FH-RIDN) to hallucinate unaligned very Low-resolution(LR) face images (32x32x3) to its 8x High-Resolution(HR) counterparts (256x256x3). Our proposed method, FH-RaGAN, consists of generator with novel arthitecture and improved discriminator. Generator is supposed to generate visually pleasant hallucinated face images, while discriminator aims to evaluate how much input images are realistic. With constant adversarial learning, FH-RaGAN is able to hallucinate perceptually plausible face images. Extensive experiments on large face datset demonstrate that the proposed method significantly outperforms the state-of-the-art.



## Proposed Method ##

In this paper, we prpose a novel GAN-based FH method, Face Hallucination based on Relativistic Average Generative Adversarial Network (FH-RaGAN).  

In summary, the contributions of this paper are mainly in five aspects:

1. Our proposed method is capable of hallucinating a unaligned tiny (i.e. 32x32x3) LR face image to hallucinated version (256x256x3) with an ultra upscaling factor 8x.

2. we present a Residual in Internel Dense Block, which boosts the flow of feature through the generator and provide abundant feature for the hallucination process.

3. we exploit relativistic average discriminator (RaD), which evaluates the probability that the given HR images is more realistic than super-resolved images. 

4. Contrary to the classical FH methods, our method does not involve any prior information or claim facial landmark points for its hallucinating, which facilitate the whole training process and add model robustness

5. we compare with state-of-the-art methods, and demonstrate that our method outerforms these approaches regarding both qualitatively and quantitatively.



In this section, our proposed method FH-RaGAN will be presented in detail. FH-RaGAN  mainly consists of two parts: novel network architecture and improved loss function. We present FH-RaGAN architecutre first and then discuss the improvement of the discriminator and perceptual loss.



## Architecture of GAFH-RIDN ##

![ICASSP_Overview (1)](https://github.com/KennethXiang/End-to-End-Generative-Adversarial-Face-Hallucination-Through-Residual-In-Internal-Dense-Network/blob/master/ICASSP_Overview%20(1).png)

Figure 1.  The architecture of our Face Hallucination Relativistic Average Generative Adversarial Network (FH-RaGAN). $I_{HR}$ and $I_{FH}$ denote HR face images and face hallucinated images respectively. DNB describes Dense Nested Block, as shown in Fig. 2; K, n, s represent kernel size, number of feature maps and strides correspondingly.



![ICASSP_RIDB](https://github.com/KennethXiang/End-to-End-Generative-Adversarial-Face-Hallucination-Through-Residual-In-Internal-Dense-Network/blob/master/icassp_RIDB.png)

​	**Figure 2. ** **Top**: Dense Nested Block(DNB) stacked by multiple RIDB. **Bottom**: the architecture of our proposed Residual in Internel Dense Block (RIDB).



#### Loss Function

The total loss function $L_{perceptual}$ can be represented as two parts:   $L_{content}$ and $L_{adversarial}$. For the propose of improving the fidelity of the face images, we introduce the $L_{content}$ . In addition, an adversarial loss $L_{adversarial}$ is expected to enhance perceptual quality from visual aspect. The formulas are as follows, Content loss: sharp visually appealing results

$$ L_{perceptual} = \alpha L_{content} + \beta L_{G}^{adv}$$																															 (3)

Where $\alpha$ and $\beta$ were corresponding fixed parameters used to balance $L_{content}$ and $L_{adversarial}$. We set $\alpha$ = 1 ,$\beta$ = $10^{-3}$ respectively.



## Experiments

### Datasets

We conducted experiments on public large-scale face image datasets, CelebA, most of which are frontal facial images. It consists of 202,599 face images of 10,177 celebrities. In our experiments, 162,048 HR face images are randomly selected as the training set, and the next 40,511 images were used as the testing set. 

### Comparisons with state-of-the-Art Methods

The quantitative comparison among state-of-the-art methods were demonstrated in Table 1. It shown the average PSNR and SSIM criterion of these methods on the CelebA dataset for scale factor 8x. 

#### Quantitative Comparison

![quantitative_compar.png](https://github.com/KennethXiang/End-to-End-Generative-Adversarial-Face-Hallucination-Through-Residual-In-Internal-Dense-Network/blob/master/quantitative_compar.png)

**Table1. Quantitative comparison on CelebA dataset for scale factor x8 in terms of average PSNR(db) and SSIM. Numerical  in bold are indicated the best evaluation results among state-of-the-art methods.**



Qualitative results were depicted in Fig5. and Fig6.

#### Qualitative Comparison



![icassp_4x_result](https://github.com/KennethXiang/End-to-End-Generative-Adversarial-Face-Hallucination-Through-Residual-In-Internal-Dense-Network/blob/master/icassp_4x_result.png)

**Fig5. Qualitative comparison of 4x super-resolved face images on CelebA dataset. Column: (a) HR ground truth; (b) Input LR; (c) Bicubic (d) SRGAN (e) ESRGAN (f) Our method.** 



![icassp_8x_result](https://github.com/KennethXiang/End-to-End-Generative-Adversarial-Face-Hallucination-Through-Residual-In-Internal-Dense-Network/blob/master/icassp_8x_result.png)

**Fig6. Qualitative comparison of 8x super-resolved face images on CelebA dataset. Column: (a) HR ground truth; (b) Input LR; (c) Bicubic (d) SRGAN (e) ESRGAN (f) Our method.** 



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
