# Variational AE
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

## Data Information
<a href="https://drive.google.com/drive/folders/1a8PUZcW1PeDsrF43sWFsVxsCaniQEzbn?usp=sharing" target="_blank">Repository</a> for the MNIST dataset. <a href="https://drive.google.com/drive/folders/1bsV12MWuyU7GMA-fr8vcPiayjUOFEOgt?usp=sharing">Folder</a> for the trained VAE model and visualizations.

Dictionary of parameters: https://github.com/JerryLiuMY/VAE/blob/main/params/params.py

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
