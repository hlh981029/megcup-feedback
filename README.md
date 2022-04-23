# MegCup 2022 Team Feedback

## Members
- [Ling-Hao Han](https://github.com/hlh981029)
- [Zuo-Liang Zhu](https://github.com/NK-CS-ZZL)
- [Weilei Wen](https://github.com/wwlCape)

Our team (FeedBack) achieve the first place in [MegCup  raw image denoising competiton]([比赛 | MegStudio (brainpp.com)](https://studio.brainpp.com/competition/5?tab=rank)) ! This is the available code.

## Environment

```bash
conda env create -f environment.yaml
conda activate feedback
```

## Dataset Preparation

You can download the [dataset](https://studio.brainpp.com/competition/5?tab=questions) to `data`. Please modify `options/feedback.yaml` to set  data path, pretrained weights path …

```bash
|--data
   |--competition_train_input.0.2.bin
   |--competition_train_gt.0.2.bin
   |--competition_test_input.0.2.bin
```

## Evaluation

```bash
# evaluate on dataset
# log and config file will be saved to ./output/feedback
git clone https://github.com/hlh981029/megcup-feedback.git
cd megcup-feedback
python test.py

# generate result bin file
# result will be saved to ./output/feedback/submit/model_best_result.bin
python test.py --submit
```



## Acknowledgement

This project is based on [[G2L-search](https://github.com/ShangHua-Gao/G2L-search)],[[Swin-transformer](https://github.com/microsoft/Swin-Transformer)],[[Restormer](https://github.com/swz30/Restormer)],[[BasicSR](https://github.com/xinntao/BasicSR)].

