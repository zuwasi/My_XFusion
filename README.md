# XFusion
Deep learning-based spatiotemporal fusion for high-fidelity ultra-high-speed x-ray radiography  
A model to reconstruct high quality x-ray images by combining the high spatial resolution of high-speed camera and high temporal resolution of ultra-high-speed camera image sequences.  

## Prerequisites
This implementation is based on the [BasicSR toolbox](https://github.com/XPixelGroup/BasicSR). Data for model pre-training are collected from the [REDS dataset](https://seungjunnah.github.io/Datasets/reds).  

## Usage
### Package description
Currently, xfusion supports 2 model familities for high-quality xray image sequence reconstruction- the EDVR and Swin vision transformer, respectively.

### Package installation
Navigate to the project root directory and then run
```
pip install .
```
to install the package to the selected virtual environment.

### Initialization
Run
<pre>
xfusion init --model_type <i>[EDVRModel or SwinIRModel]</i>
</pre>
After initialization, a configuration file "xfusion.conf" will be generated in the home directory. This configuration file will be updated automatically within the workflow of the xfusion package.

### Data preparation
#### Data for model pretraining
Download the Sharp dataset called [train_sharp](https://drive.google.com/open?id=1YLksKtMhd2mWyVSkvhDaDLWSc1qYNCz-) and Low Resolution dataset called [train_sharp_bicubic](https://drive.google.com/open?id=1a4PrjqT-hShvY9IyJm3sPF0ZaXyrCozR) from [REDS dataset](https://seungjunnah.github.io/Datasets/reds) to the directories specified in the "convert" section of the configuration file.
#### Data for model fine tuning
Fine tuning data are not available at this moment.
#### Data for testing
There are two sets of sample data to be downloaded from the [Tomobank](https://tomobank.readthedocs.io/en/latest/source/data/docs.data.radio.html).

### Data conversion
To convert the REDS data to gray-scale, run
<pre>
xfusion convert --dir-lo-convert <i>[directory/to/low resolution/RGB/training image]</i> --dir-hi-convert <i>[directory/to/high resolution/RGB/training image]</i> --out-dir-lo <i>[directory/to/low resolution/gray-scale/training image]</i> --out-dir-hi <i>[directory/to/high resolution/gray-scale/training image]</i>
</pre>

### Training
Run
<pre>
xfusion train --dir-lo-train <i>[directory/to/low resolution/gray-scale/training image]</i> --dir-hi-train <i>[directory/to/high resolution/gray-scale/training image]</i> --dir-lo-val <i>[directory/to/low resolution/gray-scale/validation image]</i> --dir-hi-val <i>[directory/ti/high resolution/gray-scale/validation image]</i> --opt <i>directory/to/training setting/yaml file</i> --path-train-meta-info-file <i>[directory/to/training image/meta data]</i> --path-val-meta-info-file <i>[directory/to/validation image/meta data]</i> --pretrain_network_g <i>[directory/to/model weight/file/for/model initialization]</i>
</pre>

### Test data
To download test data, run
<pre>
xfusion download --dir-inf <i>[tomobank/link/address/of/test/dataset]</i> --out-dir-inf <i>[directory/to/testing image]</i>
</pre>

### Inference
Run
<pre>
xfusion inference --opt <i>directory/to/testing dataset/setting/yaml file</i> --arch-opt <i>directory/to/training setting/yaml file</i> --model_file <i>[path/to/model file]</i> --machine <i>tomo or polaris</i>
</pre>
Currently to work for EDVRModel in the single-process mode and SwinIRModel in the multi-process mode.

## Ultra-Lite Notes
**Repository Structure:**
```
XFusion/
├── xfusion/                    # Main package
│   ├── __main__.py            # CLI entry point (init, train, inference, download, convert)
│   ├── config.py              # Configuration management
│   ├── utils.py               # Utilities (YAML, image processing)
│   ├── inference/             # Inference pipeline
│   │   ├── infer.py           # EDVR model inference
│   │   ├── infer_swin_ddp.py  # SwinIR distributed inference
│   │   ├── model/             # Model architectures
│   │   │   └── edvr_models.py # EDVR implementation with attention
│   │   ├── dataset/           # Data loading
│   │   └── ops/               # Operations (deformable convolution)
│   └── train/                 # Training pipeline
│       └── basicsr/           # BasicSR framework integration
├── setup.py                   # Package setup
├── envs/                      # Environment configs
└── docs/                      # Documentation
```

**Core Dependencies (Inference Path):**
- torch (neural networks)
- torchvision (image utilities)
- numpy (numerical operations)
- opencv-python (image processing)
- Pillow (image I/O)
- PyYAML (config files)
- scikit-image (metrics: SSIM)
- pandas (results export)
- tqdm (progress bars)
- natsort (natural sorting)

**Quick Start (Ultra-Lite):**
1. `bash scripts/dev_install.sh` - Setup environment
2. `python xfusion_lite.py validate` - Check installation
3. `python xfusion_lite.py infer-synthetic` - Run synthetic inference
4. `pytest -v tests/` - Run tests

**Models:** EDVR (Enhanced Deformable Video Restoration) and SwinIR (Swin Transformer) for spatiotemporal fusion of x-ray sequences.

**Performance Baseline (CPU):**
- Synthetic forward pass (before optimization): 1208.42 ms (Python 3.13, CPU)
- Synthetic forward pass (after optimization): 1110.23 ms (Python 3.13, CPU)
- **Improvement: 98.19 ms (8.1% faster)** via torch.inference_mode() + cudnn.benchmark
