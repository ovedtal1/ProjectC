# Deep-Learning Method for Low-Field MRI Reconstruction with priors learned from reference high-field data


PyTorch implementation of ViT-Fuser + Hybrid loss for Low-Field MRI reconstruction

<h1 align="center">

  <img src="https://github.com/user-attachments/assets/55c9ddec-7f11-4b14-8145-b3556cc218e2" height="400">
</h1>
  <p align="center">
    <a href="https://github.com/ovedtal1">Tal Oved</a> •
  </h1>
  <p align="center">
    <a href="https://il.linkedin.com/in/tal-oved-75b46b242">Linkedin</a> •
  </p>


- [Table of contents](#Table-of-contents)
  * [Background](#background)
  * [Dataset](#Dataset)
  * [Quick start](#Quick-start)
  * [Future Work](#Future-Work)
  * [References](#references)


## Background
Despite its promise, low-field MRI adoption is limited by prolonged scan times and low
SNR, restricting clinical viability. Leveraging reference high-field MRI scans for this task is a novel
approach, and to our knowledge, the first of its kind.
We introduce ViT-Fuser, a method for improving low-field MRI reconstruction and its clinical
value.
The ViT-Fuser, a transformer-based method, boosts SNR in low-field MRI by leveraging reference
high-field scans. A novel ’Hybrid’ loss combining SSIM and Feature-style losses ensures sharper
details and better texture preservation.


## Dataset
We used the 'LUMIERE' dataset

## Quick start

- Clone the repo:
```console
git clone https://github.com/ovedtal1/ProjectC.git
```
- Install requirements.txt
- Download the train data to registered_data folder
- Download test data to test_data folder
- Train the models with the Train*.ipynb files
- Test the results with the Test*.ipynb files

## Future Work
- Explore Additional Loss Functions
- Explore for Enhanced Feature Fusion Techniques
- Extend Testing Across Diverse Datasets

## References
* Koonjoo, N., Zhu, B., Bagnall, G.C. et al. Boosting the signal-to-noise of low-field MRI with deep learning image reconstruction. Sci Rep 11, 8248 (2021). https://doi.org/10.1038/s41598-021-87482-7
* Arnold TC, Freeman CW, Litt B, Stein JM. Low-field MRI: Clinical promise and challenges. J Magn Reson Imaging. 2023 Jan;57(1):25-44. doi: 10.1002/jmri.28408. Epub 2022 Sep 19. PMID: 36120962; PMCID: PMC9771987.
* Kevin N. Sheth, et al. "Assessment of Brain Injury Using Portable, Low-Field Magnetic Resonance Imaging at the Bedside of Critically Ill Patients." JAMA Neurol. 2021
* Patricia M. Johnson and Yvonne W. Lui  “The deep route to low-field MRI with high potential” ,Nature, 2023 Nov.
* M. Lustig, D. L. Donoho, J. M. Santos and J. M. Pauly, "Compressed Sensing MRI," in IEEE Signal Processing Magazine, vol. 25, no. 2, pp. 72-82, March 2008, doi: 10.1109/MSP.2007.914728.
* H. K. Aggarwal, M. P. Mani and M. Jacob, "MoDL: Model-Based Deep Learning Architecture for Inverse Problems," in IEEE Transactions on Medical Imaging, vol. 38, no. 2, pp. 394-405, Feb. 2019, doi: 10.1109/TMI.2018.2865356.
* Kang Lin, Reinhard Heckel , “Vision Transformers Enable Fast and Robust Accelerated MRI”, in PMLR 172:774-795, 2022.
* Lau V, Xiao L, Zhao Y, Su S, Ding Y, Man C, Wang X, Tsang A, Cao P, Lau GKK, Leung GKK, Leong ATL, Wu EX. Pushing the limits of low-cost ultra-low-field MRI by dual-acquisition deep learning 3D superresolution. Magn Reson Med. 2023 Aug;90(2):400-416.
* Suter, Y., Knecht, U., Valenzuela, W. et al. The LUMIERE dataset: Longitudinal Glioblastoma MRI with expert RANO evaluation. Sci Data 9, 768 (2022). https://doi.org/10.1038/s41597-022-01881-7
* H. Zhao, O. Gallo, I. Frosio and J. Kautz, "Loss Functions for Image Restoration With Neural Networks," in IEEE Transactions on Computational Imaging, vol. 3, no. 1, pp. 47-57, March 2017, doi: 10.1109/TCI.2016.2644865.
* Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14. Springer International Publishing, 2016.‏
* Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter. “GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium”, NIPS, 2017.
* Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, Sanjiv Kumar. "Rethinking FID: Towards a Better Evaluation Metric for Image Generation", CVPR, 2024. 

