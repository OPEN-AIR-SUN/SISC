# Semi-supervised Implicit Scene Completion from Sparse LiDAR

[**New Repo**](https://github.com/AIR-DISCOVER/LODE) 

[**Paper**](https://arxiv.org/pdf/2111.14798.pdf) 

Created by Pengfei Li, Yongliang Shi, Tianyu Liu, Hao Zhao, Guyue Zhou and YA-QIN ZHANG from <a href="http://air.tsinghua.edu.cn/EN/" target="_blank">Institute for AI Industry Research(AIR), Tsinghua University</a>.

![demo](doc/demo.gif)

For complete video, click [HERE](https://youtu.be/8x_XOSrr5K0).

We use the proposed model trained on the KITTI dataset to predict implicit completion results on the novel [DAIR-V2X](http://air.tsinghua.edu.cn/dair-v2x/) dataset. The results are impressive:

![china](doc/v2x.jpg)
![china_1](doc/v2x.gif)



![teaser](doc/qualitative.png)

![sup0](doc/qualitative_0.png)

![sup1](doc/qualitative_1.png)

![sup2](doc/qualitative_2.png)

![sup3](doc/qualitative_3.png)

![sup4](doc/qualitative_4.png)


## Introduction

Recent advances show that semi-supervised implicit representation learning can be achieved through physical constraints like Eikonal equations. However, this scheme has not yet been successfully used for LiDAR point cloud data, due to its spatially varying sparsity. 

In this repository, we develop a novel formulation that conditions the semi-supervised implicit function on localized shape embeddings. It exploits the strong representation learning power of sparse convolutional networks to generate shape-aware dense feature volumes, while still allows semi-supervised signed distance function learning without knowing its exact values at free space. With extensive quantitative and qualitative results, we demonstrate intrinsic properties of this new learning system and its usefulness in real-world road scenes. Notably, we improve IoU from 26.3\% to 51.0\% on SemanticKITTI. Moreover, we explore two paradigms to integrate semantic label predictions, achieving implicit semantic completion. Codes and data are publicly available.

## Citation

If you find our work useful in your research, please consider citing:

    @misc{li2021semisupervised,
        title={Semi-supervised Implicit Scene Completion from Sparse LiDAR}, 
        author={Pengfei Li and Yongliang Shi and Tianyu Liu and Hao Zhao and Guyue Zhou and Ya-Qin Zhang},
        year={2021},
        eprint={2111.14798},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

## Installation

### Requirements
    
    CUDA=11.1
    python>=3.8
    Pytorch>=1.8
    numpy
    ninja
    MinkowskiEngine
    tensorboard
    pyyaml
    configargparse
    scripy
    open3d
    h5py
    plyfile
    scikit-image



Clone the repository:
    
    git clone https://github.com/OPEN-AIR-SUN/SISC.git


### Data preparation

Download the SemanticKITTI dataset from 
[HERE](http://semantic-kitti.org/assets/data_odometry_voxels.zip). Unzip it into the same directory as `SISC`.



## Training and inference
The configuration for training/inference is stored in `opt.yaml`, which can be modified as needed.

### Scene Completion

Run the following command for a certain `task` (train/valid/visualize):

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_sc.py --task=[task] --experiment_name=[experiment_name]


### Semantic Scene Completion
#### SSC option A
Run the following command for a certain `task` (ssc_pretrain/ssc_valid/train/valid/visualize):

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_ssc_a.py --task=[task] --experiment_name=[experiment_name]

Here, use ssc_pretrain/ssc_valid to train/validate the SSC part. Then the pre-trained model can be used to further train the whole model.

#### SSC option B
Run the following command for a certain `task` (train/valid/visualize):

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_ssc_b.py --task=[task] --experiment_name=[experiment_name]


## Model Zoo
Our pre-trained models can be downloaded here:
<table border="2">
    <tr>
        <td style="background-color:green"><center><b>Ablation</td> 
        <td style="background-color:green" colspan="6"><center><b>Pretrained Checkpoints</td> 
   </tr>
    <tr>
        <td><b><center>data augmentation</td>    
        <td width="150">
            <a href="https://drive.google.com/file/d/1emXd-yTPfBf2gBmnggIANxCDg73mnPn5/view?usp=sharing">
                    <center>no aug
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1a1TzSgBwPs_IKkkaS2CdUa4hrmiKYq4_/view?usp=sharing">
                    <center>rotate & flip
            </a>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
    </tr>
    <tr>
        <td><b><center>Dnet input</td>    
        <td>
            <a href="https://drive.google.com/file/d/1GwWAHlHkg--07UzPq37nyCzh0Mnx__55/view?usp=sharing">
                    <center>radial distance
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1IHzXW6DjaYYvr76vtPPEr6sspojx8-Ba/view?usp=sharing">
                    <center>radial distance & height
            </a>
        </td> 
        <td>
        </td>  
        <td>
        </td> 
        <td>
        </td> 
        <td>
        </td> 
    </tr>
    <tr>
        <td><b><center>Dnet structure</td>    
        <td>
            <a href="https://drive.google.com/file/d/1jwHxrRH5xaW95MgaiQ1lYTg8l57E6Taj/view?usp=sharing">
                    <center>last1 pruning
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1ydzQx4loYYkICJKJi20YG6t05Osb3Djr/view?usp=sharing">
                    <center>last2 pruning
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1jrugPHXjrv7z5PUQxF_rM-yFGjeHZD_8/view?usp=sharing">
                    <center>last3 pruning
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1TpkvCEtRGls3ZJklyOiDYwKoUGEH4cCZ/view?usp=sharing">
                    <center>last4 pruning
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1S59qwyUgl14vAC-Ri8jZxKAK2B50bKnt/view?usp=sharing">
                    <center>Dnet relu
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1A6_wyJHVZRHudwtaD5w5HebCiv_oL7bY/view?usp=sharing">
                    <center>4convs output
            </a>
        </td>  
    </tr>
    <tr>
        <td><b><center>Gnet structure</td>    
        <td>
            <a href="https://drive.google.com/file/d/19vX4i773A6Df6YLTdyP_MxzoR8KCX1Gf/view?usp=sharing">
                    <center>width128 depth4
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1SED3cV4Fc6Sf2F3bIaf8l5KwgkqI6RMu/view?usp=sharing">
                    <center>width512 depth4
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1s1WBgNhr_gImO-wDNNGqwcHOziRjTXh5/view?usp=sharing">
                    <center>width256 depth3
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1-rVag5fkg3l1WzvyjS4zpKZQBowkXj7p/view?usp=sharing">
                    <center>width256 depth5
            </a>
        </td> 
        <td>
            <a href="https://drive.google.com/file/d/1IW6wUFTej_wBwzSWwFQe5iOT5KXke2Pm/view?usp=sharing">
                    <center>Gnet relu
            </a>
        </td>
        <td>
        </td> 
    </tr>
    <tr>
        <td><b><center>point sample</td>    
        <td>
            <a href="https://drive.google.com/file/d/1qBx3ZKAwRhdZ-BvFsJcUZI1MqPBUli26/view?usp=sharing">
                    <center>on:off=1:2
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1DzxctLzRBmm_W23O2Jum9kqSZGQ-JMXp/view?usp=sharing">
                    <center>on:off=2:3
            </a>
        </td>
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
    </tr>
    <tr>
        <td><b><center>positional encoding</td>    
        <td>
            <a href="https://drive.google.com/file/d/1MTiB5BgrSMj0tEmz7UykVcUKGgkOJr0J/view?usp=sharing">
                    <center>no encoding
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/12Eoyb1ClU75F_p37wyssVD7INJy2KlHO/view?usp=sharing">
                    <center>incF level10
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1j46UUuLoRT-8eH6VlyNbJEbU3SRU3oEY/view?usp=sharing">
                    <center>incT level5
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1RLl_OjhrdSnqtXL88-Q1hXszEtBd-gVD/view?usp=sharing">
                    <center>incT level15
            </a>
        </td>      
        <td>
        </td>      
        <td>
        </td>      
    </tr>
    <tr>
        <td><b><center>sample strategy</td>    
        <td>
            <a href="https://drive.google.com/file/d/1RQgA_NAuNcBCXDtHTgEatkBme7GfumLG/view?usp=sharing">
                    <center>nearest
            </a>
        </td>     
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
    </tr>
    <tr>
        <td><b><center>scale size</td>    
        <td>
            <a href="https://drive.google.com/file/d/1hJb4woXN5uuG7WKOKgwvLzkWxC-Smh5L/view?usp=sharing">
                    <center>scale 2
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/18YPYrKvC7KcMp0nLqU98WnjJs6JTKsda/view?usp=sharing">
                    <center> <b> <u> scale 4
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1D8DLWcGFxrFR5_RtrNlV1-Ov7-JPIdTT/view?usp=sharing">
                    <center>scale 8
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1lAhTYSJQmdAdTWcIpCAHkbOb4UBItMgf/view?usp=sharing">
                    <center>scale 16
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1tNrCnqmcb8_xgBEL5elis36E3yrSrMLv/view?usp=sharing">
                    <center>scale 32
            </a>
        </td>  
        <td>
        </td>  
    </tr>
    <tr>
        <td><b><center>shape size</td>    
        <td>
            <a href="https://drive.google.com/file/d/1iM2xVFh1Qk27HMKhKp5WyjkSoxsAJqoI/view?usp=sharing">
                    <center>shape 128
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1QDngtgrYjoMlk4ZKi6bODH8XJnj1aN0N/view?usp=sharing">
                    <center>shape 512
            </a>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
    </tr>
    <tr>
        <td><b><center>SSC</td>    
        <td>
            <a href="https://drive.google.com/file/d/17e5M2Z-TFcplfL61b54Zea8lCrBylqyT/view?usp=sharing">
                    <center>SSC option A
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1eecCo4_fyuOcfn2zTidSq07xYRrWpfjN/view?usp=sharing">
                    <center>SSC option B
            </a>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
    </tr>
    <tr>
        <td><b><center>Seg-Net&SSC-Net for SSC-A</td>    
        <td>
            <a href="https://drive.google.com/file/d/1zDUgd-NSpwaOQ4vKH-K-r_yF66n1OSYj/view?usp=sharing">
                    <center>ssc_pretrain
            </a>
        </td>     
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
    </tr>
</table>


These models correspond to the ablation study in our paper. The `Scale 4` works as our baseline.

<table border="2">
    <tr>
        <td style="background-color:green"><center><b>Ablation</td> 
        <td style="background-color:green" colspan="6"><center><b>Corresponding Configs</td> 
   </tr>
    <tr>
        <td><b><center>data augmentation</td>    
        <td width="150">
            <a href="https://drive.google.com/file/d/1CDpWMqX5KGqIQBpboA-9DGPlLDbNxE03/view?usp=sharing">
                    <center>no aug
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/18sO3s3eGnnabxI3a4yA4ijb1VQ5jteJB/view?usp=sharing">
                    <center>rotate & flip
            </a>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
    </tr>
    <tr>
        <td><b><center>Dnet input</td>    
        <td>
            <a href="https://drive.google.com/file/d/1UU0gc7s-DEncDhFWEGUmPJo9XnaoJZ4N/view?usp=sharing">
                    <center>radial distance
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1GaZajOquy4ZNP6Z1e9sftM2YhTLayh_E/view?usp=sharing">
                    <center>radial distance & height
            </a>
        </td> 
        <td>
        </td>  
        <td>
        </td> 
        <td>
        </td> 
        <td>
        </td> 
    </tr>
    <tr>
        <td><b><center>Dnet structure</td>    
        <td>
            <a href="https://drive.google.com/file/d/1iOtUKn3RTBvnvtj9COsqyYVZrrA-r0q-/view?usp=sharing">
                    <center>last1 pruning
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1VR_i1xTtToxgrdfb2OniXjgnbZQrpOLD/view?usp=sharing">
                    <center>last2 pruning
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1iCE_6cQasoBZa5SSww4Ke1Ky1B1Ytl7o/view?usp=sharing">
                    <center>last3 pruning
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1-4p306ZsYpuKAgUFeLaZLmoiojKvCIrv/view?usp=sharing">
                    <center>last4 pruning
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1sDNYFfazWwEkmKDxQnEDviKQNEapwzUV/view?usp=sharing">
                    <center>Dnet relu
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1l_fbaHmaGu19QR7iwgJLJsAEj1KNVsJg/view?usp=sharing">
                    <center>4convs output
            </a>
        </td>  
    </tr>
    <tr>
        <td><b><center>Gnet structure</td>    
        <td>
            <a href="https://drive.google.com/file/d/1WQ0zYloUFpJkOnugasgI8JUwoiFJTHbT/view?usp=sharing">
                    <center>width128 depth4
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1LbZT85TJKYuvljr4KEEwxPmW8TOdD-UD/view?usp=sharing">
                    <center>width512 depth4
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1_4o5t4LJKx5A0j7Oc6YgLQNDMhA3old6/view?usp=sharing">
                    <center>width256 depth3
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1hb6ObirbJKUQTfws1wRPdTuVhMYDg5cC/view?usp=sharing">
                    <center>width256 depth5
            </a>
        </td> 
        <td>
            <a href="https://drive.google.com/file/d/1_fElZNPrxRlnG1mk88E-FbZAqV9Bj6Xr/view?usp=sharing">
                    <center>Gnet relu
            </a>
        </td>
        <td>
        </td> 
    </tr>
    <tr>
        <td><b><center>point sample</td>    
        <td>
            <a href="https://drive.google.com/file/d/1Q6F4od-4aMcJk98V-Yt20POYx3hDkwIU/view?usp=sharing">
                    <center>on:off=1:2
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1CBYpkw3ras7dt5XKioaDFABinNTuOgZK/view?usp=sharing">
                    <center>on:off=2:3
            </a>
        </td>
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
    </tr>
    <tr>
        <td><b><center>positional encoding</td>    
        <td>
            <a href="https://drive.google.com/file/d/1DF-_Kizocc9dyArAYgRrjgjWXUZgDVpX/view?usp=sharing">
                    <center>no encoding
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1ksFSRjd80mM71SqiymojsxK87-kdzXt6/view?usp=sharing">
                    <center>incF level10
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1Q8ChxbWU2mXRPv1BMwviX8javh03YmvF/view?usp=sharing">
                    <center>incT level5
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1paafCI0b8ZmOUnxwtFEPqEyI0HcsHk_f/view?usp=sharing">
                    <center>incT level15
            </a>
        </td>      
        <td>
        </td>      
        <td>
        </td>      
    </tr>
    <tr>
        <td><b><center>sample strategy</td>    
        <td>
            <a href="https://drive.google.com/file/d/1BSoA7Veg3Y_lMpkwFUcc3SVafP5qyrpa/view?usp=sharing">
                    <center>nearest
            </a>
        </td>     
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
    </tr>
    <tr>
        <td><b><center>scale size</td>    
        <td>
            <a href="https://drive.google.com/file/d/1LYmhl1HfT1YYVbIpxnugvOeNKWaA95QL/view?usp=sharing">
                    <center>scale 2
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1P_mZcXCdme6BFWBTCpJCNQcKb4Ye5A83/view?usp=sharing">
                    <center> <b> <u> scale 4
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1vAVno6ZKBwqqSkwhPRnh_f-YtOhbN16g/view?usp=sharing">
                    <center>scale 8
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1YZK9irwZSBAmCIq6rNCgn8cHrpWor-Jr/view?usp=sharing">
                    <center>scale 16
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1xkYE_xm--LuT-BRsR92ti1YytmVo2rY_/view?usp=sharing">
                    <center>scale 32
            </a>
        </td>  
        <td>
        </td>  
    </tr>
    <tr>
        <td><b><center>shape size</td>    
        <td>
            <a href="https://drive.google.com/file/d/1lDxWYxLlwP1guHxIkebRieGUOdU9fbg2/view?usp=sharing">
                    <center>shape 128
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1rltYF8TnzuqmvwAC3a_nzYE4sXyHAlL8/view?usp=sharing">
                    <center>shape 512
            </a>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
    </tr>
    <tr>
        <td><b><center>SSC</td>    
        <td>
            <a href="https://drive.google.com/file/d/1agpi1v3tfDzMXq0vzMTbd1lXUYQz1NYT/view?usp=sharing">
                    <center>SSC option A
            </a>
        </td>  
        <td>
            <a href="https://drive.google.com/file/d/1WJq8_e298APLKdiY1xolJQcziWTX3yu4/view?usp=sharing">
                    <center>SSC option B
            </a>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
        <td>
        </td>  
    </tr>
</table>

