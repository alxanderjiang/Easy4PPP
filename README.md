# Easy4PPP
[[中]](./README-zh.md) &ensp; [[EN]](./README.md)

An Easily Applied and Recompiled  Multi-platform Precise Point Position (PPP) Toolbox Coded in Python
## Quick Start
1. we provide a set of example data and a quick-start jupyter notebook tutorial "ppp.ipynb". Make sure that you have sucessufully installed the Python as well as the numpy in your environment.
2. Due to the file size limitation of github (no more than 25MB for a file), we compress the "data" folder into zip file. In order to try the provided example of GPS precise point position (PPP), you need first unzip the "data.zip" into the "data" folder in the project path. 
3. After unzipping the "data" folder, run all the blocks of "ppp.ipynb", you all get an Easy4PPP solution log file in form of ".npy".
4. We provided an ecample of visualizing the solution log file. Run all the blocks of nav_result.ipynb, you can get figures about the PPP convergence curve, the STEC scatter and the residuls scatter plot.
## Downloading and preperations
1. Download the **zip pakeage directly** or using git clone by running the following commend:
```bash
git clone https://github.com/alxanderjiang/Easy4PPP.git
```
2. Unzip the sample data folder: data.zip to the same path of Easy4PPP. If linux but no GUI, please run the following commends:
```bash
cd Easy4PPP
unzip data.zip
```
3. Ensure that the numpy, tqdm and ipykernel is available in your Python environment. If not, please run the following commends to install:
```bash
pip install numpy
pip install tqdm
pip install ipykernel
```
numpy and tqdm is used in the core codes while ipykernel is necessary to run Jupyter Notebook tutorials.
## Running
1. Get the sample results by running the main function of sppp.py or running the ppp.ipynb in the Jupyter Notebook.
2. After processing, you will get a solution log named as "jfng1320.240.out.npy" in format of numpy array in the folder: nav_result.
3. Running the nav_result.ipynb, you will get six pictures of PPP results and products. The details are listed in UserMannual.pdf.
