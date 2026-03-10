# DICycleGAN
Image Dehazing Algorithm Based on Unsupervised Learning.

# env
u can use `pip install -r requirements.txt` to install some necessary dependency lib.

# train
traning use ubuntu22.04.

The following directory needs to be created.

`checkpoints`,

`data/test/A`,`data/test/B`,

`data/train/A`,`data/train/B`,

`data/val/A`,`data/val/B`

`logs/model`,`logs/train`

The `test`, `train`, and `val` folders under `data` correspond to the test dataset, training dataset, and validation dataset, respectively. Within each dataset folder, folder A contains foggy images, and folder B contains fog-free images.

The `train` folder under `logs` stores the TensorBoard visualization data generated during training.

The `checkpoints` folder stores the discriminator and generator models saved during training.
