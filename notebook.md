Date: 1/15

Notes:

- Single optimizer AdamW

PRE TRAINING
step: 19992/20000 | loss: 3.1131 | lr 6.0000e-05 | norm 0.2090 | time: 364.71ms | tok-sec: 1437535.06 | total time: 122.92m | eta: 0.0m | total tokens: 10,481,565,696
step: 19993/20000 | loss: 3.0737 | lr 6.0000e-05 | norm 0.2006 | time: 364.69ms | tok-sec: 1437626.22 | total time: 122.93m | eta: 0.0m | total tokens: 10,482,089,984
step: 19994/20000 | loss: 3.1280 | lr 6.0000e-05 | norm 0.1970 | time: 364.63ms | tok-sec: 1437850.88 | total time: 122.93m | eta: 0.0m | total tokens: 10,482,614,272
step: 19995/20000 | loss: 3.1338 | lr 6.0000e-05 | norm 0.2061 | time: 364.68ms | tok-sec: 1437684.49 | total time: 122.94m | eta: 0.0m | total tokens: 10,483,138,560
step: 19996/20000 | loss: 3.0771 | lr 6.0000e-05 | norm 0.1931 | time: 364.40ms | tok-sec: 1438772.82 | total time: 122.95m | eta: 0.0m | total tokens: 10,483,662,848
step: 19997/20000 | loss: 3.0381 | lr 6.0000e-05 | norm 0.2031 | time: 364.86ms | tok-sec: 1436943.27 | total time: 122.95m | eta: 0.0m | total tokens: 10,484,187,136
step: 19998/20000 | loss: 3.0789 | lr 6.0000e-05 | norm 0.2316 | time: 364.82ms | tok-sec: 1437126.39 | total time: 122.96m | eta: 0.0m | total tokens: 10,484,711,424
Validation loss: 3.0881

MID TRAINING
step: 00992/01000 | loss: 1.8659 | norm 0.1789 | time: 455.83ms | tok-sec: 1150186.55 | total time: 7.78m | total tokens: 520,617,984
step: 00993/01000 | loss: 1.8678 | norm 0.1975 | time: 454.72ms | tok-sec: 1153002.89 | total time: 7.78m | total tokens: 521,142,272
step: 00994/01000 | loss: 1.9824 | norm 0.1950 | time: 457.60ms | tok-sec: 1145724.42 | total time: 7.79m | total tokens: 521,666,560
step: 00995/01000 | loss: 1.8223 | norm 0.1740 | time: 458.39ms | tok-sec: 1143771.58 | total time: 7.80m | total tokens: 522,190,848
step: 00996/01000 | loss: 1.8889 | norm 0.1723 | time: 455.72ms | tok-sec: 1150457.33 | total time: 7.81m | total tokens: 522,715,136
step: 00997/01000 | loss: 1.8696 | norm 0.2060 | time: 456.17ms | tok-sec: 1149335.31 | total time: 7.82m | total tokens: 523,239,424
step: 00998/01000 | loss: 1.8686 | norm 0.2139 | time: 455.53ms | tok-sec: 1150931.81 | total time: 7.82m | total tokens: 523,763,712
val/smoltalk_loss: 1.7425
val/mmlu_loss: 2.3500
val/gsm8k_loss: 1.7931

1/18

Notes:

- we are now using different learning rates for different layers but still using only adamw
- using scaled down lrs in midtraining
- using bos aligned and best fit conversation for the mid training data loader
- soft cap logits
- untied weights

PRE TRAINING
step: 19992/20000 | loss: 3.0753 | matrix lr 7.5001e-05 | norm 0.1595 | time: 374.55ms | tok-sec: 1399780.68 | total time: 126.59m | eta: 0.1m | total tokens: 10,481,565,696
step: 19993/20000 | loss: 3.0359 | matrix lr 7.5000e-05 | norm 0.1801 | time: 374.46ms | tok-sec: 1400124.70 | total time: 126.59m | eta: 0.0m | total tokens: 10,482,089,984
step: 19994/20000 | loss: 3.0895 | matrix lr 7.5000e-05 | norm 0.1628 | time: 374.26ms | tok-sec: 1400871.25 | total time: 126.60m | eta: 0.0m | total tokens: 10,482,614,272
step: 19995/20000 | loss: 3.0969 | matrix lr 7.5000e-05 | norm 0.1518 | time: 374.43ms | tok-sec: 1400215.64 | total time: 126.61m | eta: 0.0m | total tokens: 10,483,138,560
step: 19996/20000 | loss: 3.0388 | matrix lr 7.5000e-05 | norm 0.1636 | time: 374.83ms | tok-sec: 1398748.74 | total time: 126.61m | eta: 0.0m | total tokens: 10,483,662,848
step: 19997/20000 | loss: 2.9989 | matrix lr 7.5000e-05 | norm 0.1715 | time: 374.63ms | tok-sec: 1399492.05 | total time: 126.62m | eta: 0.0m | total tokens: 10,484,187,136
step: 19998/20000 | loss: 3.0447 | matrix lr 7.5000e-05 | norm 0.1823 | time: 375.17ms | tok-sec: 1397474.95 | total time: 126.63m | eta: 0.0m | total tokens: 10,484,711,424
Validation loss: 3.0485

MID TRAINING
step: 00992/01000 | loss: 1.3055 | norm 0.1703 | time: 404.60ms | tok-sec: 1295829.85 | total time: 6.87m | total tokens: 520,617,984
step: 00993/01000 | loss: 1.3065 | norm 0.1754 | time: 400.57ms | tok-sec: 1308852.04 | total time: 6.88m | total tokens: 521,142,272
step: 00994/01000 | loss: 1.3191 | norm 0.1636 | time: 403.69ms | tok-sec: 1298752.62 | total time: 6.89m | total tokens: 521,666,560
step: 00995/01000 | loss: 1.2863 | norm 0.1567 | time: 397.95ms | tok-sec: 1317482.52 | total time: 6.89m | total tokens: 522,190,848
step: 00996/01000 | loss: 1.3134 | norm 0.1559 | time: 405.30ms | tok-sec: 1293584.94 | total time: 6.90m | total tokens: 522,715,136
step: 00997/01000 | loss: 1.2932 | norm 0.1461 | time: 402.69ms | tok-sec: 1301978.38 | total time: 6.91m | total tokens: 523,239,424
step: 00998/01000 | loss: 1.3277 | norm 0.1539 | time: 404.49ms | tok-sec: 1296175.85 | total time: 6.91m | total tokens: 523,763,712
val/smoltalk_loss: 1.5777
val/mmlu_loss: 2.8233
val/gsm8k_loss: 1.6708

Future notes

- we are now matching d20 depth. This goes up a lot to around 500M model params
- effectively disabled GQA
- using bos aligned and best fit conversation for the mid training data loader
