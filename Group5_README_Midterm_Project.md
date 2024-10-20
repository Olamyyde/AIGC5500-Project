
## Midterm Project - Deep Learning Optimizers (AIGC 5500)

### Overview
This project compares the performance of three popular deep learning optimizers: **Adam**, **RMSprop**, and **AdamW**. The comparison is done using a fully connected neural network trained on the **KMNIST dataset**, which consists of handwritten Japanese characters.

### Setup Instructions

#### 1. Clone the Repository:
First, clone the project repository (if applicable):
```
git clone <repo_url>
cd <repo_directory>
```

#### 2. Create the Conda Environment:
Create a Conda environment using the provided `environment.yml` file or manually by running:
```
conda create --name dl_optimizer python=3.8 pytorch torchvision torchaudio jupyter matplotlib scikit-learn -c pytorch
conda activate dl_optimizer
```

#### 3. Install Required Libraries:
If you haven't installed the necessary dependencies, run:
```
pip install -r requirements.txt
```

#### 4. Dataset Preparation:
The project uses the **KMNIST dataset**, which can be directly loaded via PyTorch:
```
from torchvision.datasets import KMNIST
train_dataset = KMNIST(root='./data', train=True, download=True)
test_dataset = KMNIST(root='./data', train=False, download=True)
```

#### 5. Running the Code:
To execute the Jupyter notebook or Python scripts, simply launch JupyterLab or Jupyter Notebook:
```
jupyter notebook Group5_Midterm_project.ipynb
```
Make sure to follow the instructions in the notebook to select the optimizer and adjust hyperparameters as needed.

### Project Structure:
- **Group5_Midterm_project.ipynb**: The main Jupyter notebook containing the code for training the model and comparing optimizers.
- **results/**: Contains graphs and tabular data comparing the performance of the optimizers.
- **README.md**: This file with setup and run instructions.
- **requirements.txt**: A list of Python libraries and dependencies needed for the project.

