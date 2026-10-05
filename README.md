# ML From Scratch: 33 Machine Learning Algorithms, One at a Time

Companion code for the ML series on the YouTube channel (https://www.youtube.com/@enzochangtw) (videos in Mandarin, code in English).

Each episode covers one algorithm, from intuition to math to strengths and weaknesses to code. Every folder runs on your own machine. Episodes are ordered so that each one builds on the ones before it, so going in order is the easiest path.

## Setup

Requires Python 3.9+.

```bash
git clone <this-repo-url>
cd ml-from-scratch
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Seasons 1–3 only need NumPy, Matplotlib and scikit-learn. Later seasons will add PyTorch (Season 4) and Gymnasium (Season 5). Those will be listed in `requirements.txt` when their episodes are released.

## How each episode's code is organized

1. Generate or load data, then split into training and test sets
2. Implement the algorithm from scratch with NumPy (or PyTorch for deep learning)
3. Do the same with a standard library for comparison
4. Compare results on the test set and plot them
5. The folder's README lists experiments and challenges to try

## Roadmap

Status: **Available** means the folder exists and runs. **Planned** means the episode hasn't been released yet, and the folder name may change.

### Season 0: Foundations

| Ep | Topic | Key ideas | Status |
| --- | --- | --- | --- |
| 0 | What does machine learning actually learn? | Supervised / unsupervised / reinforcement learning, loss, gradient descent, overfitting, train/test split | Video only |

### Season 1: Classic Supervised Learning

| Ep | Algorithm | Good at | Watch out for | Folder | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | Linear Regression | Simple, fast, interpretable | Only fits straight lines; sensitive to outliers | [`ep01_linear_regression`](ep01_linear_regression) | Available |
| 2 | Ridge & Lasso | Reduce overfitting; Lasso also selects features | You have to tune the penalty strength | `ep02_ridge_lasso` | Planned |
| 3 | Logistic Regression | Gives probabilities; strong baseline classifier | Linear decision boundary | `ep03_logistic_regression` | Planned |
| 4 | k-Nearest Neighbors | No training step; intuitive | Slow predictions; struggles with many features | `ep04_knn` | Planned |
| 5 | Naive Bayes | Very fast; great for text | Assumes features are independent | `ep05_naive_bayes` | Planned |
| 6 | Support Vector Machine | Strong on small data; kernels handle curves | Slow on large data; kernel choice is tricky | `ep06_svm` | Planned |
| 7 | Decision Tree | Very interpretable; little preprocessing | Overfits easily; unstable | `ep07_decision_tree` | Planned |
| 8 | Random Forest | Robust and accurate | Slower; harder to interpret | `ep08_random_forest` | Planned |
| 9 | AdaBoost | Turns weak learners into a strong one | Sensitive to noise and outliers | `ep09_adaboost` | Planned |
| 10 | Gradient Boosting | Very accurate on tabular data | Trains sequentially, so it's slow; many hyperparameters | `ep10_gradient_boosting` | Planned |
| 11 | XGBoost / LightGBM / CatBoost | Wins most tabular competitions | Can overfit; needs tuning | `ep11_xgboost_lightgbm_catboost` | Planned |
| 12 | Linear Discriminant Analysis | Classifies and reduces dimensions at once | Assumes Gaussian classes with equal covariance | `ep12_lda` | Planned |

### Season 2: Unsupervised Learning

| Ep | Algorithm | Good at | Watch out for | Folder | Status |
| --- | --- | --- | --- | --- | --- |
| 13 | K-Means | Simple; scales well | You must pick K; assumes round clusters | `ep13_kmeans` | Planned |
| 14 | Hierarchical Clustering | No K needed; readable dendrogram | Slow on large data | `ep14_hierarchical_clustering` | Planned |
| 15 | DBSCAN | Finds clusters of any shape; marks noise | Sensitive to `eps`; struggles with varying density | `ep15_dbscan` | Planned |
| 16 | Gaussian Mixture Models + EM | Soft clustering with probabilities | Can get stuck in a local optimum | `ep16_gmm_em` | Planned |
| 17 | Principal Component Analysis | Reduces dimensions and noise | Linear only; components can be hard to interpret | `ep17_pca` | Planned |
| 18 | t-SNE & UMAP | Excellent 2D/3D visualizations | Visualization only; distances between clusters can mislead | `ep18_tsne_umap` | Planned |
| 19 | Isolation Forest | Fast anomaly detection | Weaker in very high dimensions | `ep19_isolation_forest` | Planned |

### Season 3: Probabilistic and Sequential Models

| Ep | Algorithm | Good at | Watch out for | Folder | Status |
| --- | --- | --- | --- | --- | --- |
| 20 | Hidden Markov Model | Modeling hidden states over time | Markov assumption is limiting | `ep20_hmm` | Planned |
| 21 | Kalman Filter | Optimal tracking for linear-Gaussian systems | Needs EKF/UKF variants for nonlinear systems | `ep21_kalman_filter` | Planned |
| 22 | Gaussian Process | Predictions with uncertainty | Cost grows with the cube of the data size | `ep22_gaussian_process` | Planned |

### Season 4: Neural Networks and Deep Learning

| Ep | Algorithm | Good at | Watch out for | Folder | Status |
| --- | --- | --- | --- | --- | --- |
| 23 | Perceptron → MLP + Backpropagation | Can approximate almost any function | Needs lots of data; hard to interpret | `ep23_mlp_backprop` | Planned |
| 24 | Convolutional Neural Network | The standard for images | Needs heavy compute and labeled data | `ep24_cnn` | Planned |
| 25 | RNN & LSTM | Sequences and time series | Vanishing gradients; can't run in parallel | `ep25_rnn_lstm` | Planned |
| 26 | Autoencoder & VAE | Compression and generation without labels | VAE outputs tend to be blurry | `ep26_autoencoder_vae` | Planned |
| 27 | Generative Adversarial Network | Sharp generated images | Unstable training; mode collapse | `ep27_gan` | Planned |
| 28 | Transformer & Attention | Long-range context; parallel; the basis of LLMs | Cost grows with the square of sequence length | `ep28_transformer` | Planned |
| 29 | Diffusion Models | State-of-the-art image generation | Slow, step-by-step sampling | `ep29_diffusion` | Planned |
| 30 | Graph Neural Network | Data shaped like graphs (roads, molecules, social networks) | Over-smoothing; hard to scale | `ep30_gnn` | Planned |

### Season 5: Reinforcement Learning

| Ep | Algorithm | Good at | Watch out for | Folder | Status |
| --- | --- | --- | --- | --- | --- |
| 31 | Q-Learning | Simple; no model of the environment needed | Only works with small state spaces | `ep31_q_learning` | Planned |
| 32 | Deep Q-Network | Handles large state spaces | Unstable; needs many samples | `ep32_dqn` | Planned |
| 33 | Policy Gradient & PPO | Continuous actions; PPO is stable | Very sample-hungry; reward design is hard | `ep33_policy_gradient_ppo` | Planned |

## Contributing

Found a bug, or have a clearer way to explain something? Issues and pull requests are welcome.
