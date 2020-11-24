# End-to-End Generative Adversarial Face Hallucination Through Residual in Internel Dense Network

### Abstract

Face hallucination has been a highly attractive computer vision research topic in recent years. It is still a particularly challenging task since the human face has a complex and delicate structures. In this paper, we propose the novel network structure, namely end-to-end Generative Adversarial Face Hallucination through Residual in Internal Dense Network (GAFH-RIDN), to hallucinate an unaligned tiny (32x32 pixels) low-resolution face image to its 8x (256x256 pixels) high-resolution counterpart. We propose a new architecture called Residual in Internal Dense Block (RIDB) for the generator and exploit an improved discriminator, Relativistic average Discriminator (RaD). In GAFH-RIDN, the generator is used to generate visually pleasant hallucinated face images, while the improved discriminator aims to evaluate how much input images are realistic. With continual adversarial learning, GAFH-RIDN is able to hallucinate perceptually plausible face images. Extensive experiments on large face datasets demonstrate that the proposed method significantly outperforms other state-of-the-art methods.

![ICASSP_Overview_4](/Users/wangxiang/Code/Github_Repository/End-to-End-Generative-Adversarial-Face-Hallucination-Through-Residual-In-Internal-Dense-Network/ICASSP_Overview_4.png)



**Fig. 1.** The architecture of our end-to-end Generative Adversarial Face Hallucination through Residual in Internal Dense Network (GAFH-RIDN). $I_{HF}$ represents HF image. $I_{HR}$ and $I_{LR}$ denote HR and LR face image respectively. K, n, and s represent kernel size, the number of feature maps and strides respectively. SFM is the Shallow Feature Module. MDBM describes the Multi-level Dense Block Module. UM is the Upsampling Module. DNB represents the Dense Nested Block as shown in Fig. 2.

## Proposed Method ##

In this paper, we propose a novel GAN-based FH method, end-to-end Generative Adversarial Face Hallucination through Residual in Internal Dense Network (GAFH-RIDN), as shown in Fig. 1. The contributions of this paper are mainly in four aspects: 

In summary, the contributions of this paper are mainly in five aspects:

1. Our proposed method is capable of hallucinating an LR (32x32 pixels) unaligned tiny face image to a Hallucinated Face (HF) image (256x256 pixels) with an ultra upscaling factor 8x.

2. We propose the Residual in Internal Dense Block (RIDB), which boosts the flow of features through the generator and provides hierarchical features for the hallucination process.

3. We exploit the Relativistic average Discriminator (RaD), which evaluates the probability that the given HR face images are more realistic than HF images.

4. Contrary to classical face hallucination methods, our method does not involve any prior information or claim facial landmark points for its hallucinating, which facilitates the whole training process and enhances the model robustness.

5. we compare with state-of-the-art methods, and demonstrate that our method outerforms these approaches regarding both qualitatively and quantitatively.

## Proposed Method

In this section, we will first describe the proposed architecture and demonstrate the Residual in Internal Dense Block (RIDB). Next, we will discuss the improved discriminator. Finally, we will present the perceptual loss function used in GAFH-RIDN. The architecture of GAFH-RIDN is shown in Fig. 1.



![ICASSP_RIDB](/Users/wangxiang/Code/Github_Repository/End-to-End-Generative-Adversarial-Face-Hallucination-Through-Residual-In-Internal-Dense-Network/ICASSP_RIDB.png)

**Fig. 2.  Top**: Dense Nested Block (DNB) composed of multiple RIDBs. **Bottom:** The architecture of our proposed Residual in Internal Dense Block (RIDB).



### Network Architecture

As shown at the top of Fig. 1, the proposed generator mainly consists of three stages: Shallow Feature Module (SFM), Multi-level Dense Block Module (MDBM), and Upsampling Module (UM). The LR face image $I_{LR}$ is fed into the SFM as the initial input. At the end, hallucinated face image $I_{HF}$ is obtained from the UM. As for the SFM, we utilize one convolutional (Conv) layer to extract the shallow feature maps. It can be expressed as follows:


$F_{SFM}=f_{Conv}(I_{LR})$

where $f_{Conv}$ represents the Conv operation in the SFM. $F_{SFM}$ denotes the shallow (low-level) features and serves as the input to the MDBM. The following module MDBM is built up by multiple Dense Nested Blocks (DNB) formed by several RIDBs, which will be discussed in the next subsection. The  procedure of high-level feature extraction in MDBM can be formulated as:

$F_{MDBM} = f_{DNB,i}(f_{DNB,i-1}(\cdot \cdot \cdot(f_{DNB,1}(F_{SFM}))\cdot \cdot \cdot))$

where $f_{DNB,i}$ denotes  high-level feature extraction of the $i$-th DNB, $F_{MDBM}$ represents the high-level feature extracted by MDBM. As for each DNB, it includes 3 RIDBs cascaded by residual connections and one scale layer, as shown in Fig. 2. It can be formulated as:

$F_{DNB,i} =\alpha F_{i,j}(F_{i,j-1}(\cdot \cdot \cdot F_{i,1}(F_{DNB,i-1})\cdot \cdot \cdot))+F_{DNB,i-1}$

where $F_{DNB,i-1}$, $F_{DNB,i}$ denotes the input and output of $i$-th DNB, $F_{i,j}$ represents the $j$-th RIDB of the $i$-th DNB. We assign $\alpha$ to be 0.2 in the scale layer. Next, the low-level and high-level features should be fused to boost hallucination performance via skip connection. Let $F_{fused}$ denotes the fused feature, the feature fusion process can be expressed as:

$F_{fused}=f_{Conv}(F_{MDBM})+F_{SFM}$

Furthermore, the fused feature $F_{fused}$ is passed to the UM followed by one Conv layer. And then, the fused feature is transformed from the LR space to the HR space through upsampling layers in the UM. The hallucination process can be formulated as:

$I_{HF}=f_{UM}(F_{fused})=H_{GAFH-RIDN}(I_{LR})$

where $f_{UM}$ represents the upsampling operation in the UM, $H_{GAFH-RIDN}$ denotes the function of our GAFH-RIDN. Finally, we obtain the HF image $I_{HF}$.



### Residual in Internal Dense Block

As mentioned in Sec.1, we propose a novel architecture RIDB for the generator, which is used to form the DNB (as shown in Fig. 2). The proposed RIDB is able to extract hierarchical features and address the vanishing-gradient problem, which is the commonly encountered issue in SRGAN, ESRGAN, SRDenseNet, URDGN, RDN. The proposed RIDB is made up of four internal dense blocks and all the internal dense blocks are cascaded through residual connections performing identity mapping. The architecture of the RIDB is expressed as:

$F_{RIDB,p} = F_{p,q}(F_{p,q-1}(\cdot \cdot F_{p,1}(F_{RIDB,p-1})\cdot \cdot))+F_{RIDB,p-1}$

where $F_{RIDB,p-1}$ and $F_{RIDB,p}$ denote the input and output of the $p$-th RIDB respectively, $F_{p,q}$ represents the $q$-th internal dense block of $p$-th RIDB. In addition, an internal dense block is a composition of two groups of the Conv layer followed by the LeakyReLU activation layer. And the two groups are linked by dense skip connections. Each internal dense block can be calculated as follows:

$F_{q,k} =\delta(W_{q,k}[F_{q,k=1},F_{q,k=2}])$

where $F_{q,k}$ represents the output of $k$-th Conv layer of $q$-th internal dense block. $[F_{q,k=1},F_{q,k=2}]$ refers to the concatenation of feature maps in $q$-th internal dense block. $W_{q,k}$ is the weights of the $k$-th Conv layer. $\delta$ denotes the LeakyReLU activation. By involving residual learning and more dense connections in the RIDB, the feature maps of each layer are propagated into all succeeding layers, promoting an effective way to extract hierarchical features. Thus, our proposed method is capable of obtaining abundant hierarchical feature information and alleviating the vanishing-gradient problem.

### Improved Discriminator

Instead of using the discriminator of Standard GAN (SGAN), inspired by RaGAN, we adopt the Relativistic average Discriminator (RaD) in our method. Thanks to RaD, the discriminator of GAFH-RIDN has the ability to distinguish how the given HR face image is more authentic than the hallucinated face image. The architecture of our discriminator is shown at the bottom of Fig. 1. The limitation of the SGAN in GAN, SRGAN, URDGN is that they only concentrate on increasing the probability that fake samples belong to real rather than decreasing the probability that real samples belong to real simultaneously. In other words, the standard discriminator ignores real samples during the learning procedure RaGAN, as a result, the model can not provide sufficient gradients when updating the generator, which causes the problem of gradient vanishing for training generator. The standard discriminator can be expressed as: 

$D(x) = \sigma (C(x))$ 

where $x$ can be either $I_{HR}$ or $I_{HF}$ in this context, $\sigma$ represents the sigmoid function, and $C(x)$ denotes the output of non-transformed discriminator. 

As Eq. 8 shows, the standard discriminator only evulates the probability for a given real sample or a generated sample. According to RaGAN, RaD takes into consideration that how a given real sample is more authentic than a given generated sample. The RaD can be formulated as:

$D(x_{r},x_{f}) = \sigma (C(x_{r}) - E_{x_{f}}[C(x_{f})])$

where $E_{x_{f}}$ denotes the average of the fake samples in one batch. Contrary to standard discriminator, as Eq. 9 shows, the probability predicted by RaD relies on both real sample $x_{r}$ and fake sample $x_{f}$, which is capable of making discriminator relativistic. In our GAFH-RIDN, we can optimize the RaD by $L_{D}^{adv}$ based on Eq. 10, and the generator is updated by $L_{G}^{adv}$, as in Eq. 11.

$L_{D}^{adv} = -\mathbb{E}_{I_{HR}}\sim p_{(I_{HR})}\left [ log\left ( D(I_{HR},I_{HF}\right ))\right ]-\mathbb{E}_{I_{HF}}\sim p_{(I_{HF})}\left [ log\left( 1-D\left ( I_{HF},I_{HR} \right )\right )\right ]$



$L_{G}^{adv}=-\mathbb{E}_{I_{HR}}\sim p_{(I_{HR})}\left [ log\left (1- D(I_{HR},I_{HF}\right ))\right ]-\mathbb{E}_{I_{HF}}\sim p_{(I_{HF})}\left [ log\left(D\left ( I_{HF},I_{HR} \right )\right )\right ]$

where $I_{HR}$ and $I_{HF}$ denote HR images and HF images respectively, $D(\cdot)$ describes the probability predicted by RaD, $\mathbb{E}$ represents the expectation, $I_{HR} \sim P_{I_{HR}}$ and $I_{HF} \sim P_{I_{HF}}$ represents the HR images distribution and HF images distribution respectively. Because of this property, our proposed method is capable of allowing the probability of $I_{HR}$ being real to decrease while letting the probability of $I_{HF}$ being real increase and benefitting from gradients of both $I_{HR}$ and $I_{HF}$ in the adversarial training. 

Therefore our proposed method can address the gradient vanishing problem. Our discriminator contains 9 Conv layers with the number of 3x3 kernels and the stride of 1 or 2 alternately. The channels of feature maps increase by a factor 2, from 64 to 512. The resulting 512 feature maps are passed through two dense layers. Finally, after the sigmoid activation layer, RaD estimates the probability that the given HR face images are more realistic than HF images.

### Content Loss

Taking advantage of content loss $L_{content}$ is able to promote ulteriorly detail enhancement. We adopt the pre-trained VGG-19 as the feature extractor to obtain feature representation used to calculate $L_{content}$. We extract low-level feature maps of HR and HF images obtained by the $3^{rd}$ Conv layer before the $4^{th}$ maxpooling layer respectively. HR and HF feature maps are defined as $\phi_{3,4}$. $L_{content}$ is defined as follows:

$L_{content} = \frac{1}{WH}\sum_{h=1}^{H}\sum_{w=1}^{W}\left \|\phi_{3,4} (I_{HR})-\phi_{3,4} (I_{HF}) \right \|^{2}$

where $\phi$ represents the feature extractor and $W,H$ denote the dimensions of feature maps. 

### Perceptual Loss

The perceptual loss function $L_{perceptual}$ for generator can be represented as two parts: $L_{content}$ and $L_{G}^{adv}$. We introduce the content loss to improve the fidelity of the HF image. In addition, an adversarial loss is expected to enhance perceptual quality of the HF image from the visual aspect. The formula is as follows:

$L_{perceptual} = \alpha L_{content} + \beta L_{G}^{adv}$

where $\alpha$ and $\beta$ are corresponding hyper-parameters used to balance $L_{content}$ and $L_{G}^{adv}$. We empirically set $\alpha$ = 1, $\beta$ = $10^{-3}$ respectively.

## Experiments

In this section, we will first present the details of datasets and implementation. Next, we will discuss the comparisons with the state-of-the-art methods [VDSR,CBN,SRGAN,FSRFCH,TDN,ESRGAN] qualitatively and quantitatively. 

### Implementation Details

We conducted experiments on the large-scale face image dataset, CelebA. It consists of 202,599 face images of 10,177 celebrities. We randomly selected 162,048 HR face images as the training set, and the next 40,511 images were used as the testing set. We cropped the HR face images and resized them to 256x256 pixels, and then obtained LR (32x32 pixels) input images by downsampling HR images using bicubic interpolation with a downsampling factor of 8x. In the proposed generator, we set the number of DNBs to 4, and totally 12 RIDBs were used. In the training phase, we trained the proposed method for 10000 epochs and the training batch size was set to 8. We used Adam with $\beta_{1}$ = 0.9, $\beta_{2}$ = 0.999 to optimize the proposed method. The learning rate was set to $10^{-4}$. We alternately updated the generator and discriminator.

### Comparisons with State-of-the-art Methods

In the experiments, we compared the proposed method with the state-of-the-art methods [TDN,VDSR,CBN,SRGAN,FSRFCH,ESRGAN] qualitatively and quantitatively.

![ICASSP_Table1](/Users/wangxiang/Code/Github_Repository/End-to-End-Generative-Adversarial-Face-Hallucination-Through-Residual-In-Internal-Dense-Network/ICASSP_Table1.png)

**Table 1.** Quantitative comparison on CelebA dataset for scaling factor 8x, in terms of average PSNR(dB) and SSIM. Numbers in bold are the best evaluation results among state-of-the-art methods.



**Qualitative Comparison**

Qualitative results among these methods are shown in Fig. 3. We observe that the bicubic interpolation produces heavy blur and fails to generate clear textures. For SRGAN, it outputs noticeable artifacts around facial components, especially in the eyes, nose, and mouth regions. In particular, ESRGAN, produces unrealistic textures and involves severe ghosting artifacts. In contrast, it is obvious that our proposed method is capable of producing visually pleasant and authentic HF images. 

![icassp_8x_result](/Users/wangxiang/Code/Github_Repository/End-to-End-Generative-Adversarial-Face-Hallucination-Through-Residual-In-Internal-Dense-Network/icassp_8x_result.png)

**Fig. 3. ** Comparison of visual results with state-of-the-art methods on scaling factor 8x. (a) HR images, (b) LR inputs, (c) Bicubic interpolation, (d) Results of SRGAN, (e) Results of ESRGAN, and (f) Our results



**Quantitative Comparison**

Table 1 shows the quantitative comparison on 8x HF images. The results demonstrate that our proposed method achieves the best performance among all methods. Especially, our method produces the highest score at 24.28dB/0.71 in terms of PSNR and SSIM respectively. Furthermore, compared with the second-best FSRFCH, 23.14dB/0.68, our method outperformes it with a large margin of 1.14dB/0.03. The performance proves the effectiveness of the proposed RIDB and the optimized RaD used in our method.

## Conclusions

In this paper, we proposed a novel end-to-end face hallucination method (GAFH-RIDN) to hallucinate a tiny LR (32x32 pixels) unaligned face image to its 8x HR (256x256 pixels) version. By exploiting Residual in Internal Dense Block (RIDB) and Relativistic average Discriminator (RaD), our method successfully produced photo-realistic hallucinated face images. Extensive experiments demonstrated that GAFH-RIDN was superior to the state-of-the-art methods on the face benchmark qualitatively and quantitatively. 

