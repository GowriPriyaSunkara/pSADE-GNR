# p-Laplacian GNN + GMVAE + VAE

This repository implements a graph-based machine learning pipeline for predicting body composition metrics (ALM, BMD, BFP) using anthropometric data.

The project integrates:

📊 Graph Neural Networks (p-Laplacian GNN)

🔄 Variational Autoencoders (VAE)

🔀 Gaussian Mixture VAEs (GMVAE)

📈 Correlation-weighted graph construction

🌐 k-NN graph learning

📉 Regression evaluation (RMSE, MAE, R²)

This work explores how geometry + topology + deep learning can improve prediction of biological measurements.

We model the dataset as a graph:

Nodes → individuals
Edges → similarity (k-NN graph)
Edge weights → optionally correlation-based

Then apply:

p-Laplacian diffusion for nonlinear smoothing
Latent representations (VAE / GMVAE)
Graph-based regression (PGNN)

Structure

├── PGNN_L2.py              # Base PGNN (L2 graph, no weights)

├── PGNN_L2_cc.py          # PGNN with correlation edge weights

├── PGNN_VAE.py            # VAE + PGNN 

├── PGNN_VAE_cc.py         # VAE + PGNN + correlation edge weights

├── PGNN_GMVAE_L2.py       # GMVAE + PGNN 

├── PGNN_GMVAE_L2_cc.py    # GMVAE + PGNN + correlation edge weights

├── results/               # Output metrics, logs, plots

└── README.md