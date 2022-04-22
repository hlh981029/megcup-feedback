# MegCup 2022 Team Feedback

```bash
## prepare data
# train input: /data/competition_train_input.0.2.bin
# train gt: /data/competition_train_gt.0.2.bin
# test input: /data/competition_test_input.0.2.bin

# evaluate on train dataset
python test.py

# generate result bin file
python test.py --submit
```