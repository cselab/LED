# Multiscale Simulations of Complex Systems by Learning their Effective Dynamics

Code and data for the paper: PR Vlachas, G. Arampatzis, C. Uhler, and P Koumoutsakos, *Multiscale Simulations of Complex Systems by Learning their Effective Dynamics*, Nature Machine Intelligence, 2022.

# Demontration on the FitzHugh Nagumo Equation (FHN)

The scripts to generate the training, validation, and test data for each application can be found in the ./LED/Data folder.
Run these scripts in the respective order, i.e. for the FHN equation:
```
python3 0_data_gen.py
python3 1_creating_figures.py
python3 2_create_training_data.py
python3 3_data_gen_test.py
python3 4_create_test_data.py
```
Once the data are generated, navigate to ./LED/Code/Experiments/FHN/Local and run any of the scripts.
| Script  | Description |
| ------------- | ------------- |
| 0_PCA.sh  | Training and testing the dimensionality reduction with PCA/DiffMaps.  |
| 1_PCA_RC.sh  | Training and testing a Reservoir Computer with PCA/DiffMaps on the latent space (and multiscale testing)   |
| 2_PCA_SINDy.sh  | PCA/DiffMaps + SINDy   |
| 3_PCA_RNN.sh  | PCA/DiffMaps + RNN (LSTM/GRU)   |
| 4_AE.sh  | Training a Convolutional Autoencoder (CNN)   |
| 5_AE_RNN.sh  | LED (CNN+LSTM)   |
| 5_AE_RC.sh  | LED (AE+RC)   |

These scripts train and test the respective networks or dimensionality reduction methods, and generate plots and files with diagnostics in the ./LED/Results folder.

# Data availability

Scripts to generate the data for the FHN and the KS equations are provided in the ./Data folder.
Data for the Navier-Stokes flow past a cylinder have been generated using the in-house software library, [CubismUP-2D](https://github.com/novatig/CubismUP_2D) / [Cubism-AMR](https://arxiv.org/abs/2206.07345).
Due to the large data size, the data for this application are not uploaded here.


# Dependencies

1. The code has been tested in python 3.8 & python3.9. Create virtual environment
```
python3 -m venv venv-led
```
2. Activate virtual environment
```
source ./venv-led/bin/activate
```
3. Install dependencies
```
pip install -U pip
pip install -r requirements.txt
```




