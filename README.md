# rl30_final



Follow the [VimoRAG](https://github.com/WalkerMitty/VimoRAG) repository to download the environment

# Dataset and Models

## Stage 0
Download the humanml3d (ground truth), and our preference dataset.
[dataset](https://drive.google.com/file/d/1NAVgcWhnEJRtM__1CUV7mJzb9iXzl-cF/view?usp=sharing)
[humanml3d](https://drive.google.com/file/d/1_PUxrTur45HfU0_n4oHlTXy-UtaNzdl2/view?usp=sharing)

The file structure should look like
```bash
unzip preference_data.zip
unzip humanml3d.zip
cp -r humanml3d VimoRAG/McDPO/
cp -r preference_data InstructMotion/

cd VimoRAG/McDPO
```

The file structure should look like
```bash
preference_data/
VimoRAG/
VimoRAG/McDPO/humanml3d
InstructMotion/
```

## Stage 1 (Our models)
- Download the dataset and models from [HuggingFace](https://huggingface.co/datasets/Haidong2/VimoRAG) (or [ModelScope](https://modelscope.cn/models/Walkerhai/VimoRAG)) and put them in ``data/``
- Run

```shell
cd data/VimoRAG
unzip McDPO.zip
unzip resources.zip
mv McDPO/checkpoints {project_root}/VimoRAG/McDPO
mv output {project_root}/VimoRAG/
mv resources {project_root}/VimoRAG/
```

## Stage 2 (Pretrained models)
- Download the following three models — [InternVideo2](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4), [CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336), and [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) — and place them in the `./resources/playground` directory. The resulting file structure should be as follows:

```
├── playground
│   ├── InternVideo2-Stage2_1B-224p-f4
│   ├── models
│   │   ├── mlp2x_gelu_clip_l14_336px
│   │   └── mlp2x_gelu_internvideo2
│   ├── openai
│   │   └── clip-vit-large-patch14-336
│   └── Phi-3-mini-4k-instruct
```

```bash
cd {project_root}
cd VimoRAG/McDPO
conda env create -f environment.yml
conda activate mcdpo
bash additional_env.sh
```

generate preference_motion.jsonl (For InstructMotion Training)

```bash
# In VimoRAG/McDPO/
python motion_text_score.py --dataset_mapping humanml3d_mapping.json --gt_motion_dir {project_root}/VimoRAG/McDPO/humanml3d/new_joint_vecs --output preference_labels.jsonl
```

Make sure you are out of VimoRAG environment
```bash
mv preference_labels.jsonl {project_root}/InstructMotion/preference_data/
conda deactivate
cd {project_root}/InstructionMotion
```

# Instruction Motion

Set up environment according to [MotionGPT](https://github.com/OpenMotionLab/MotionGPT?tab=readme-ov-file#-quick-start) setup instructions below:
```bash
conda create python=3.10.6 --name mgpt
conda activate mgpt
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
cd MotionGPT
pip install -r requirements.txt
python -m spacy download en_core_web_sm
bash prepare/download_smpl_model.sh
bash prepare/prepare_t5.sh
bash prepare/download_t2m_evaluators.sh
bash prepare/download_pretrained_models.sh
```

To train the DPO model, modify the hyperparameters and paths in `src/scripts/dpo_train.sh` and run the following command (does not support distributed training):
```bash
bash src/scripts/dpo_train.sh
```
```FLIP_RATIO``` represents the probability of swapping the chosen and rejected samples in preference pairs during training

---
```cd``` into MotionGPT first.

To evaluate without peft
```
python test.py --cfg configs/config_h3d_stage3.yaml --task t2m --checkpoint /path/to/trained_model.pt
```

To evaluate with peft:
```
python test.py --cfg configs/config_h3d_stage3.yaml --task t2m --checkpoint /path/to/trained_model.pt --peft --r 8 --lora_alpha 16 --lora_dropout 0.05 
```


## Acknowledgements

- [VimoRAG](https://github.com/WalkerMitty/VimoRAG)
- [InstructMotion](https://github.com/THU-LYJ-Lab/InstructMotion)
- [MotionGPT](https://github.com/qiqiApink/MotionGPT)
- [InternVideo](https://github.com/OpenGVLab/InternVideo)
- [VideoGPT-plus](https://github.com/mbzuai-oryx/VideoGPT-plus)
- [LLaVA-Hound-DPO](https://github.com/RifleZhang/LLaVA-Hound-DPO)