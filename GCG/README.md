# llm-review-watermark
Detecting academic paper reviews generated solely using LLMs.

## Installation
Follow the instructions below to set up the environment for the experiments.

1. Install Anaconda:
    - Download .sh installer file from https://www.anaconda.com/products/distribution
    - Run: 
        ```
        bash Anaconda3-2023.03-Linux-x86_64.sh
        ```
2. Set up conda environment `rev-wm` with required packages:
    ```
    conda env create -f env.yml
    ```
3. Activate environment:
    ```
    conda activate rev-wm
    ```

### Manually Build Environment (Optional)
If setting up the environment using `env.yml` does not work, manually build an environment
with the required packages using the following steps:

1. Create Conda Environment with Python:
    ```
    conda create -n [env] python=3.10
    ```
2. Activate environment:
    ```
    conda activate [env]
    ```
3. Install PyTorch with CUDA from: https://pytorch.org/
	```
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```
    <!-- conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -->
4. Install transformers from Huggingface:
    ```
    pip install transformers
    ```
5. Install Huggingface datasets:
    ```
    pip install datasets
    ```
6. Install `accelerate`:
    ```
    pip install accelerate
    ```
7. Install `seaborn`:
    ```
    conda install anaconda::seaborn
    ```
8. Install `termcolor`:
    ```
    conda install -c conda-forge termcolor
    ```
9. Install OpenAI and PyPDF2 python package:
    ```
    pip install openai PyPDF2
    ```