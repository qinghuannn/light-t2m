<!-- # Light-T2M: A Lightweight and Fast Model for Text-to-motion Generation (AAAI 2025) -->

# [Light-T2M: A Lightweight and Fast Model for Text-to-motion Generation](https://qinghuannn.github.io/light-t2m/)  (AAAI 2025)

[![arXiv](https://img.shields.io/badge/arXiv-<2412.11193>-<COLOR>.svg)](https://arxiv.org/abs/2412.11193)


The official PyTorch implementation of the paper [**"Light-T2M: A Lightweight and Fast Model for Text-to-motion Generation"**](https://arxiv.org/abs/2412.11193).


If you find this project or the paper useful in your research, please cite us:

```bibtex
@inproceedings{light-t2m,
  title={Light-T2M: A Lightweight and Fast Model for Text-to-motion Generation},
  author={Zeng, Ling-An and Huang, Guohong and Wu, Gaojie and Zheng, Wei-Shi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## Getting Started

<details>

### 1. Create Conda Environment

<details>

We tested our code using Python 3.10.14, PyTorch 2.2.2, CUDA 12.1, and NVIDIA RTX 3090 GPUs.

```bash
conda create -n light-t2m python==3.10.14
conda activate light-t2m

# install pytorch
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121


# install requirements
pip install -r requirements.txt

# install mamba
cd mamba && pip install -e .
```

</details>

### 2. Download and preprocess the datasets

<details>

#### 2.1 Download the Datasets

We conduct experiments on the HumanML3D and KIT-ML datasets. For both datasets, you can download them by following the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git).

Then, copy both datasets to our repository. For example, the file directory for HumanML3D should look like this:

```bash
./data/HumanML3D/
├── new_joint_vecs/
├── texts/
├── Mean.npy # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── Std.npy # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```

#### 2.2 Preprocess the Datasets

To speed up data loading during training, we convert the datasets into .npy files using the following commands:

```bash
python src/tools/data_preprocess.py --dataset hml3d
python src/tools/data_preprocess.py --dataset kit
```

</details>



### 3. Download Dependencies and Pretrained Models

<details>

Download and unzip dependencies from [here](https://1drv.ms/u/s!ApyE_Lf3PFl2i4NcE8mgVUN3oX9nTQ?e=345HR5).

Download and unzip pretrained models from [here](https://1drv.ms/u/s!ApyE_Lf3PFl2i4Nb_QxAif-rcumPlg?e=O82IX1).

Then, the file directory should look like this:

```bash
./
├── checkpoints
│   ├── hml3d.ckpt
│   ├── kit.ckpt
│   └── kit_new.ckpt
├── deps
│   ├── glove
│   └── t2m_guo
└── ...
```

</details>


</details>

## Training 

<details>

We train our Light-T2M model on two RTX 3090 GPUs.

- **HumanML3D**
```bash
python src/train.py trainer.devices=\"0,1\" logger=wandb data=hml3d_light_final \
    data.batch_size=128 data.repeat_dataset=5 trainer.max_epochs=600 \
    callbacks/model_checkpoint=t2m +model/lr_scheduler=cosine model.guidance_scale=4\
    model.noise_scheduler.prediction_type=sample trainer.precision=bf16-mixed 
```

- **KIT-ML**
```bash
python src/train.py trainer.devices=\"2,3\" logger=wandb data=kit_light_final \
    data.batch_size=128 data.repeat_dataset=5 trainer.max_epochs=1000 \
    callbacks/model_checkpoint=t2m +model/lr_scheduler=cosine model.guidance_scale=4\
    model.noise_scheduler.prediction_type=sample trainer.precision=bf16-mixed 
```

</details>

## Evaluation

<details>

Set ```model.metrics.enable_mm_metric``` to ```True``` to evaluate Multimodality. Setting ```model.metrics.enable_mm_metric``` to ```False``` can speed up the evaluation.

- **HumanML3D**
```bash
python src/eval.py trainer.devices=\"0,\" data=hml3d_light_final data.test_batch_size=128 \
    model=light_final  \
    model.guidance_scale=4 model.noise_scheduler.prediction_type=sample\
    model.denoiser.stage_dim=\"256\*4\" \
    ckpt_path=\"checkpoints/hml3d.ckpt\" model.metrics.enable_mm_metric=true
```

- **KIT-ML**

We have observed that the performance of our trained model may fluctuate. Additionally, when we retrained the model on the KIT-ML dataset, we achieved improved performance with a new checkpoint (checkpoints/kit_new.ckpt).

```bash
python src/eval.py trainer.devices=\"1,\" data=kit_light_final data.test_batch_size=128 \
    model=light_final \
    model.guidance_scale=4 model.noise_scheduler.prediction_type=sample\
    model.denoiser.stage_dim=\"256\*4\" \
    ckpt_path=\"checkpoints/kit.ckpt\" model.metrics.enable_mm_metric=true
# or
python src/eval.py trainer.devices=\"1,\" data=kit_light_final data.test_batch_size=128 \
    model=light_final \
    model.guidance_scale=4 model.noise_scheduler.prediction_type=sample\
    model.denoiser.stage_dim=\"256\*4\" \
    ckpt_path=\"checkpoints/kit_new.ckpt\" model.metrics.enable_mm_metric=true
```

</details>

## Evaluating Inference Time

<details>
One hundred samples randomly selected from the HumanML3D dataset are used to evaluate the inference time. The randomly selected samples are stored in ```data/random_selected_data.npy```.

```bash
CUDA_VISIBLE_DEVICES=0 python src/test_speed.py +trainer.benchmark=true model.noise_scheduler.prediction_type=sample 
```

</details>

## Motion Generation

<details>

```bash
python src/sample_motion.py device=\"0\"  \
    model.guidance_scale=4 model.noise_scheduler.prediction_type=sample\
    text="A person walking and changing their path to the left." length=100
```

</details>


## Visualization

<details>

### 1. Download Render Dependencies

Download and unzip rendering dependencies from [here](https://1drv.ms/u/s!ApyE_Lf3PFl2i4NirCSIchbqf8D6fw?e=3chRmv). Place the rendering dependencies in the ```./visual_datas/``` directory.


### 2. Install Python Dependencies

```bash
pip install imageio bpy matplotlib smplx h5py git+https://github.com/mattloper/chumpy imageio-ffmpeg
```

### 3. Visualize the Generated Motion

```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore visualize/blend_render.py --file_dir ./visual_datas/gen_joints --mode video   --down_sample 1  --motion_list gen_motion_1 gen_motion_1
```

</details>

## Citation
If you find this project or the paper useful in your research, please cite us:

```bibtex
@inproceedings{light-t2m,
  title={Light-T2M: A Lightweight and Fast Model for Text-to-motion Generation},
  author={Zeng, Ling-An and Huang, Guohong and Wu, Gaojie and Zheng, Wei-Shi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

## Acknowlegements
Thanks to all open-source projects and libraries that supported our research:

[T2M](https://github.com/EricGuo5513/text-to-motion),
[MLD](https://github.com/ChenFengYe/motion-latent-diffusion/tree/main), 
[T2M-GPT](https://github.com/Mael-zys/T2M-GPT), 
[TEMOS](https://github.com/Mathux/TEMOS),
[FLAME](https://github.com/kakaobrain/flame),
[MoMask](https://github.com/EricGuo5513/momask-codes),
[Mamba](https://github.com/state-spaces/mamba)


## License
This project is licensed under the [MIT License](https://github.com/EricGuo5513/momask-codes/tree/main?tab=MIT-1-ov-file#readme).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.

