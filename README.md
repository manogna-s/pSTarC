# pSTarC
Official code for the paper
 
 [pSTarC: Pseudo Source Guided Target Clustering for Fully Test-Time Adaptation (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Sreenivas_pSTarC_Pseudo_Source_Guided_Target_Clustering_for_Fully_Test-Time_Adaptation_WACV_2024_paper.pdf) [[Project-page]](https://manogna-s.github.io/pstarc/) [[arXiV]](https://arxiv.org/pdf/2309.00846.pdf)
 
 Manogna Sreenivas, Goirik Chakrabarty and Soma Biswas.


## Setup

### Installation

All experiments were done using PyTorch 1.13 on NVIDIA A-5000 GPU. The environment can be setup as follows:

```
conda create -n pstarc
conda activate pstarc
conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install scikit-learn matplotlib
```



### Datasets
Download the [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification), [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html), [DomainNet-126](http://ai.bu.edu/M3SDA/) datasets and place it under `data` folder as follows:

```
data
├── visda
│   ├── train
│   ├── validation
├── officehome
│   ├── Art
│   ├── Clipart
│   ├── Product
│   ├── Realworld
├── domainnet126
│   ├── real
│   ├── sketch
│   ├── clipart
│   ├── painting
```

### Checkpoints
Download the checkpoints from [here](https://drive.google.com/drive/folders/1pv7hAi04vvsWAzZkEXKwjOfB0z03HKpD?usp=share_link) and place under the directory `./weights`

```
weights
├── visda
├── officehome
├── domainnet126
```

## Test Time Adaptation

Run the following scripts:

####  VisDA

`$sh scripts/run_visda.sh`

#### Office-Home

`$sh scripts/run_officehome.sh`

#### DomainNet-126

`$sh scripts/run_domainnet126.sh`

## Citation
If you use this code your work, please cite our paper

```
@inproceedings{sreenivas2024pstarc,
  title={pSTarC: Pseudo Source Guided Target Clustering for Fully Test-Time Adaptation},
  author={Sreenivas, Manogna and Chakrabarty, Goirik and Biswas, Soma},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2702--2710},
  year={2024}
}
```