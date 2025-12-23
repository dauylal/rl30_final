# rl30_final

Download humanml3d and put in VimoRAG/McDPO, it should look like
```bash
VimoRAG/McDPO/humanml3d
```

Follow the [VimoRAG](https://github.com/WalkerMitty/VimoRAG) repository to download the environment

# Dataset and Models

## Stage 0
Download the humanml3d (ground truth), and our preference dataset.
[humanml3d] ()
[preference_dataset]()

## Stage 1 (Our models)
- Download the dataset and models from [HuggingFace](https://huggingface.co/datasets/Haidong2/VimoRAG) (or [ModelScope](https://modelscope.cn/models/Walkerhai/VimoRAG)) and put them in ``data/``
- Run

```shell
cd data/VimoRAG
unzip McDPO.zip
unzip resources.zip
mv McDPO/checkpoints /home/michael/code/rl30_final/VimoRAG/McDPO
mv output /home/michael/code/rl30_final/VimoRAG/
mv resources /home/michael/code/rl30_final/VimoRAG/
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
cd VimoRAG/McDPO
conda env create -f environment.yml
conda activate mcdpo
bash additional_env.sh
```

generate preference_motion.jsonl (For InstructMotion Training)

```bash
# In VimoRAG/McDPO/
python motion_text_score.py --dataset_mapping humanml3d_mapping.json --gt_motion_dir /home/michael/code/VimoRAG/McDPO/humanml3d/new_joint_vecs --output preference_motion.jsonl
```

Make sure you are out of VimoRAG environment
```bash
conda deactivate
cd {root}/InstructionMotion
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


## Acknowledgements

- [VimoRAG](https://github.com/WalkerMitty/VimoRAG)
- [MotionGPT](https://github.com/qiqiApink/MotionGPT)
- [InternVideo](https://github.com/OpenGVLab/InternVideo)
- [VideoGPT-plus](https://github.com/mbzuai-oryx/VideoGPT-plus)
- [LLaVA-Hound-DPO](https://github.com/RifleZhang/LLaVA-Hound-DPO)