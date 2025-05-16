# VAE-with-SSMs

This repository hosts the implementations for (1) a Gaussian Variational Autoencoder (VAE) and (2) a VAE + State-Space Model (SSM) hybrid model—both for image generation and trained on the CelebA dataset. 

**VAE-CelebA.py** implements and trains a Variational Autoencoder using an encoder-decoder architecture on 64x64 pixel images from CelebA. The code is borrowed from an open-sourced PyTorch notebook (https://github.com/Jovana-Gentic/VAE_celeba/blob/main/pytorch-vae-celeba.ipynb). The code loads CelebA images using the provided partition CSV, preprocessing the images, defines the convolutional encoder and decoder (both modeling Gaussian distributions), trains the VAE, generates samples in batch sizes of 16, and plots the final training curves and output images.

Afterwards, we also have a **VAE-losses.py** file which uses Pandas and Matplotlib to load the VAE_loss.csv file (with loss values at each step), and graphs all three loss functions—NELBO, REC, and KL—over the course of training. This script allowed us to overlay all three losses onto one graph rather than three seperate graphs, which was helpful for the class presentation visualizing the loss functions. For the SSM+VAE model, we generated three seperate graphs.

Lastly, we have **VAE-SSM.py**, which hosts the implementation code for the VAE-SSM hybrid model. 

The CelebA dataset we used can be found here: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset.

# Using this repository
0. [Optional] Create a virtual environment for this project (where you'll download all the necessary packages and dependencies) and set your Visual Studio Code environment up on a remote server which you can log into with an ssh key (training VAEs and SSMs are computationally expensive!).
1. Save the files in this repository in your associated IDE - in particular, VAE-CelebA.py and VAE-SSM.py, which host the code for training and running the two models.
2. Download the CelebA dataset from Kaggle onto your server or other coding environment: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
3. Change the file paths for "" and ""
4. In your terminal, after cd'ing into your folder with all of these files and the dataset, run "python VAE-CelebA.py" or "python VAE-SSM.py", depending on which model you want to train and run inference on.
5. Enjoy the results and share!
