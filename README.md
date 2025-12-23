# rl30_final

Download humanml3d and put in VimoRAG/McDPO, it should look like
```bash
VimoRAG/McDPO/humanml3d
```

Follow the [VimoRAG](https://github.com/WalkerMitty/VimoRAG) repository to download the environment

# Dataset and Models

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


## Acknowledgements

- [VimoRAG](https://github.com/WalkerMitty/VimoRAG)
- [MotionGPT](https://github.com/qiqiApink/MotionGPT)
- [InternVideo](https://github.com/OpenGVLab/InternVideo)
- [VideoGPT-plus](https://github.com/mbzuai-oryx/VideoGPT-plus)
- [LLaVA-Hound-DPO](https://github.com/RifleZhang/LLaVA-Hound-DPO)