1/15

## First Run on TinyStories dataset

PRE TRAINING
step: 00993/01000 | loss: 1.5566 | lr 6.0072e-05 | norm 0.2215 | time: 365.36ms | tok-sec: 1434975.08 | total time: 6.69m | eta: 0.0m | total tokens: 520,617,984
step: 00994/01000 | loss: 1.5152 | lr 6.0053e-05 | norm 0.2106 | time: 365.29ms | tok-sec: 1435278.54 | total time: 6.69m | eta: 0.0m | total tokens: 521,142,272
step: 00995/01000 | loss: 1.5544 | lr 6.0037e-05 | norm 0.2165 | time: 365.04ms | tok-sec: 1436257.21 | total time: 6.70m | eta: 0.0m | total tokens: 521,666,560
step: 00996/01000 | loss: 1.6436 | lr 6.0024e-05 | norm 0.2171 | time: 365.04ms | tok-sec: 1436256.27 | total time: 6.70m | eta: 0.0m | total tokens: 522,190,848
step: 00997/01000 | loss: 1.4939 | lr 6.0013e-05 | norm 0.2375 | time: 364.94ms | tok-sec: 1436652.25 | total time: 6.71m | eta: 0.0m | total tokens: 522,715,136
step: 00998/01000 | loss: 1.6741 | lr 6.0006e-05 | norm 0.2224 | time: 365.09ms | tok-sec: 1436033.05 | total time: 6.72m | eta: 0.0m | total tokens: 523,239,424

MID TRAINING
step: 00093/00100 | loss: 4.4305 | norm 0.6932 | time: 462.86ms | tok-sec: 1132718.40 | total time: 0.95m | total tokens: 49,283,072
step: 00094/00100 | loss: 4.3444 | norm 0.6021 | time: 467.53ms | tok-sec: 1121407.67 | total time: 0.96m | total tokens: 49,807,360
step: 00095/00100 | loss: 4.3728 | norm 0.5263 | time: 465.66ms | tok-sec: 1125894.14 | total time: 0.97m | total tokens: 50,331,648
step: 00096/00100 | loss: 4.3285 | norm 0.5686 | time: 463.45ms | tok-sec: 1131270.93 | total time: 0.98m | total tokens: 50,855,936
step: 00097/00100 | loss: 4.1567 | norm 0.6726 | time: 469.11ms | tok-sec: 1117622.70 | total time: 0.99m | total tokens: 51,380,224
step: 00098/00100 | loss: 4.3192 | norm 0.4561 | time: 466.54ms | tok-sec: 1123785.39 | total time: 0.99m | total tokens: 51,904,512
val/smoltalk_loss: 4.2324
val/mmlu_loss: 4.2997
val/gsm8k_loss: 4.6630

## Second Run with split optimizers, different learning rates,

PRE TRAINING
step: 00993/01000 | loss: 2.4450 | matrix lr 1.0012e-05 | norm 0.8118 | time: 378.83ms | tok-sec: 1383959.62 | total time: 6.88m | eta: 0.0m | total tokens: 520,617,984
step: 00994/01000 | loss: 2.3903 | matrix lr 1.0009e-05 | norm 1.0486 | time: 379.04ms | tok-sec: 1383187.47 | total time: 6.89m | eta: 0.0m | total tokens: 521,142,272
step: 00995/01000 | loss: 2.4432 | matrix lr 1.0006e-05 | norm 0.9307 | time: 378.60ms | tok-sec: 1384810.23 | total time: 6.89m | eta: 0.0m | total tokens: 521,666,560
step: 00996/01000 | loss: 2.5021 | matrix lr 1.0004e-05 | norm 0.7607 | time: 379.20ms | tok-sec: 1382603.93 | total time: 6.90m | eta: 0.0m | total tokens: 522,190,848
step: 00997/01000 | loss: 2.3783 | matrix lr 1.0002e-05 | norm 1.0146 | time: 378.72ms | tok-sec: 1384352.55 | total time: 6.91m | eta: 0.0m | total tokens: 522,715,136
step: 00998/01000 | loss: 2.5417 | matrix lr 1.0001e-05 | norm 0.9139 | time: 378.86ms | tok-sec: 1383873.39 | total time: 6.91m | eta: 0.0m | total tokens: 523,239,424

MID TRAINING
step: 00093/00100 | loss: 4.9310 | norm 0.5553 | time: 406.50ms | tok-sec: 1289764.08 | total time: 1.18m | total tokens: 49,283,072
step: 00094/00100 | loss: 4.9833 | norm 0.5021 | time: 405.96ms | tok-sec: 1291483.56 | total time: 1.19m | total tokens: 49,807,360
step: 00095/00100 | loss: 4.9046 | norm 0.4467 | time: 397.52ms | tok-sec: 1318896.15 | total time: 1.20m | total tokens: 50,331,648
step: 00096/00100 | loss: 4.9107 | norm 0.4582 | time: 399.18ms | tok-sec: 1313426.83 | total time: 1.20m | total tokens: 50,855,936
step: 00097/00100 | loss: 4.9129 | norm 0.4143 | time: 400.29ms | tok-sec: 1309760.23 | total time: 1.21m | total tokens: 51,380,224
step: 00098/00100 | loss: 4.9928 | norm 0.4057 | time: 398.08ms | tok-sec: 1317033.55 | total time: 1.22m | total tokens: 51,904,512
val/smoltalk_loss: 4.8940
val/mmlu_loss: 5.1467
val/gsm8k_loss: 5.6669

1/18

Notes:

- we are now using different learning rates for different layers but still using only adamw
- we are now matching d20 depth. This goes up a lot to around 500M model params
- using scaled down lrs in midtraining
- effectively disabled GQA
- using bos aligned and best fit conversation for the mid training data loader
