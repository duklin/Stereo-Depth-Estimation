# Stereo Depth Estimation
This project was done on Ubuntu 18.04 with NVIDIA Titan X (cuda 11.1)
## Dependencies
* `conda`
* `git`
* `cuda`
* `data_scene_flow.zip` Dataset which can be downloaded from [here](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
## Running the project
```sh
git clone https://github.com/duklin/sde.git
cd sde

# download TensorBoard Logs and checkpoints
wget --output-document logs_and_checkpoints.tar.gz https://uni-bonn.sciebo.de/s/GShVSpQ7wXsPkRL/download
tar -xvf logs_and_checkpoints.tar.gz

# Create and activate virtual environment
conda env create -n sde -f environment.yml
conda activate sde

# Preprocessing the dataset files
python -m src.dataset.preprocess --root dataset/ --kitti-archive /path/to/data_scene_flow.zip

# Start Jupyter-Lab environment and TensorBoard in separate processes
jupyter-lab --no-browser
tensorboard --logdir logs/ --samples_per_plugin images=30
```
