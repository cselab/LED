# Multiscale Simulations of Complex Systems by Learning their Effective Dynamics

Code and data for the paper: PR Vlachas, G. Arampatzis, C. Uhler, and P Koumoutsakos, *Multiscale Simulations of Complex Systems by Learning their Effective Dynamics*, Nature Machine Intelligence, 2022.


# Demontration on the FitzHugh Nagumo Equation (FHN)

The training/validation and test data are on the ./LED/Data folder.
Navigate to ./LED/Code/Experiments/FHN/Local and run any of the scripts.
| Script  | Description |
| ------------- | ------------- |
| 0_PCA.sh  | Training and testing the dimensionality reduction with PCA/DiffMaps.  |
| 1_PCA_RC.sh  | Training and testing a Reservoir Computer with PCA/DiffMaps on the latent space (and multiscale testing)   |
| 2_PCA_SINDy.sh  | PCA/DiffMaps + SINDy   |
| 3_PCA_RNN.sh  | PCA/DiffMaps + RNN (LSTM/GRU)   |
| 4_AE.sh  | Training a Convolutional Autoencoder (CNN)   |
| 5_AE_RNN.sh  | LED (CNN+LSTM)   |

These scripts train and test the respective networks or dimensionality reduction methods, and generate plots and files with diagnostics in the ./LED/Results folder.
Similar scripts are provided for the KS equation, and the Navier-Stokes flow past a cylinder.

# Dependencies (python 3.8)

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


