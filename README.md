# MegCup 2022 Team Feedback

```bash
## prepare data
# train input: ./data/competition_train_input.0.2.bin
# train gt: ./data/competition_train_gt.0.2.bin
# test input: ./data/competition_test_input.0.2.bin

# create environment
conda env create -f environment.yaml
conda activate feedback

# evaluate on train dataset
# log and config file will be saved to ./output/feedback
python test.py

# generate result bin file
# result will be saved to ./output/feedback/submit/model_best_result.bin
python test.py --submit

```