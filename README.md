# Structure-Aware Pre-Selected Neural Rendering for Light Field Reconstruction

## Requirements

- Ubuntu 20.04.3 LTS (GNU/Linux 5.11.0-40-generic x86_64)
- python==3.7.0
- pytorch==1.10.0+cu113
- numpy==1.21.4
- einops==0.5.0
- pandas==1.3.4
- opencv==3.4.2
- pillow==8.4.0
- h5py==1.10.2
- scikit-image==0.17.2
- scikit-learn==1.0.1

## The computing infrastructure

- NVIDIA RTX A5000

## Datasets

### Stanford Datasets

We used the [The (New) Stanford Light Field Archive](http://lightfield.stanford.edu) for testing.

### Synthetic Datasets

We used the [HCI](https://lightfield-analysis.uni-konstanz.de/) 4D LF benchmark for training, [HCI](https://lightfield-analysis.uni-konstanz.de/), [HCI_old](https://lightfield-analysis.uni-konstanz.de/) old [DLFD](https://github.com/JingleiSHI/FSLFDE?tab=readme-ov-file) for testing.

### Lytro camera Datasets

We used the [Stanford Lytro Light Field Archive](http://lightfields.stanford.edu/LF2016.html) for training and testing.

Please download above datasets for training and testing. 

Note, since the preprocessing of **Stanford Datasets** is more complex, including downsampling and various flips, we provide preprocessed **Stanford Datasets** via [Google Drive](https://drive.google.com/drive/folders/1fqOQnxStVHA1t2TWMTymWMuJG9k2_Z-B?usp=share_link).

## Testing with the pre-trained model

```bash
cd dcn && sh make.sh && cd ..
python test.py -i [your_datasets_path] -m ../pretrain_model/ -in 2 -out 7 -e 1 -d [Stanf/HCI/Lytro] -dn [stanford/HCI/HCI_old/DLFD/30scenes/occlusions/reflective] -imc 0
```

Use `python test.py -h` to get more helps.

## Training

```bash
python train.py -d [Syn/Lytro] -imc 1 -i [your_datasets_path] -in 2 -out 7 -e 0 -b 128 -c 64 -lr 0.001 -g 2
```

Use `python train.py -h` to get more helps.

## Application Demo

We provide a application demo that applies the proposed method to real-world applications, including 3D display and digital refocusing. By moving the mouse, users can change the viewpoints, focus regions, and depth of field. Note, all images are rendered in real-time.


```python
python test_GUI_view_sys.py
```

- Move the mouse freely to change the position of the viewpoint. 
- Press `Esc` to the digital refocusing mode. 
- Move the mouse left and right to change the depth of focus, and move the mouse up and down to change the depth of field. 
- Press `Esc` to exit.
