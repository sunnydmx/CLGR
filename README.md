# GMTSCLR: A Contrastive Learning Framework for Extracting Graph Representation from Multivariate Time Series
## Data Preparation
[METR_LA](https://github.com/chnsh/DCRNN_PyTorch) [PeMS](https://github.com/divanoresia/Traffic)
```python
unzip mts/metr-la.h5.zip -d mts/
mkdir save_pth/metr-la
```
## Train VAE Model
```python
python pretrain_vae.py --dataset_name=metr-la
```
## Train GMTSCLR Model
```python
python pretrain_gmtsclr.py --dataset_name=metr-la
```
## Subsititute embeddings and graph representations
