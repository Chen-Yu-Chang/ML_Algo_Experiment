# EP1: Linear Regression

Linear regression from scratch on a toy "house size → price" dataset, solved three ways: gradient descent, the normal equation, and scikit-learn.

Units: size is in *ping* (a Taiwanese unit, about 3.3 m²); price is in units of 10,000 NTD.

## Run

```bash
python linear_regression.py
```

Expected output (the random seed is fixed, so everyone should get the same numbers):

```
Method                       w         b    Test MSE   Test R2
--------------------------------------------------------------
Gradient descent         51.46    323.18     20167.6     0.970
Normal equation          51.46    323.18     20167.6     0.970
scikit-learn             51.46    323.18     20167.6     0.970
```

It also saves `ep01_result.png`: the data and fitted line on the left, the loss curve on the right.

## Break it: four experiments

| Command | What happens | Think about it |
| --- | --- | --- |
| `python linear_regression.py --lr 0.01` | Converges very slowly | Are 200 steps enough? Try `--epochs 2000` |
| `python linear_regression.py --lr 1.05` | Loss explodes | Why does it blow up right after 1? (Hint: the second derivative of the loss with respect to w) |
| `python linear_regression.py --no-scale --lr 0.0005` | w looks fine, b barely moves | Why are the slopes for w and b so different without scaling? |
| `python linear_regression.py --no-scale --lr 0.01` | Explodes at step 3 | Same learning rate as above. What changed? |

## Challenges

1. Set `NOISE_STD` to 0. Do all three methods recover exactly 52 and 300?
2. Add one outlier, say a 10-ping house selling for 5,000. How much does the line tilt?
3. Extend the model to two features (size + building age). How do the gradients change?
