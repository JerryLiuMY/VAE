# Variational AE
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

## Data Information
<a href="https://drive.google.com/drive/folders/1a8PUZcW1PeDsrF43sWFsVxsCaniQEzbn?usp=sharing" target="_blank">Repository</a> for the MNIST dataset. <a href="https://drive.google.com/drive/folders/1bsV12MWuyU7GMA-fr8vcPiayjUOFEOgt?usp=sharing">Folder</a> for the trained VAE model and visualizations.

Dictionary of parameters: https://github.com/JerryLiuMY/VAE/blob/main/params/params.py

```python
from loader.loader import load_data
from models.train import train_vae
from models.train import valid_vae

# load data and perform training & validation
dataset = "mnist"
train_loader, valid_loader, input_shape = load_data(dataset)
model, train_loss = train_vae(train_loader, input_shape)
valid_loss = valid_vae(model, valid_loader)
```

```python
from loader.loader import sort_digits
from visualization.recon import plot_recon
from visualization.sample import plot_sample
from visualization.space import plot_space
from visualization.inter import plot_inter

# create sample images
digit_set = sort_digits(valid_loader)
image_set, labels = next(iter(valid_loader))

# plot various visualizations
plot_recon(model, image_set)
plot_inter(model, digit_set, d1=3, d2=9)
plot_sample(model)
plot_space(model)
```

## Demo
### Reconstructions
Reconstruct new digits from the original digits.
![alt text](./__resources__/recon.jpg?raw=true "Title")

### Latent Space Interpolation
Generate new digits by interpolating the latent vectors of two digits `3` and `9`.
![alt text](./__resources__/inter.jpg?raw=true "Title")

### VAE Generator
Generate new digits by drawing latent vectors from the learnt normal distribution.
![alt text](./__resources__/sample.jpg?raw=true "Title")

### 2D Latent Space
![alt text](./__resources__/space.jpg?raw=true "Title")
