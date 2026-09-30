# STFTransNet

Official PyTorch implementation of

> **STFTransNet: A Transformer Based Spatial Temporal Fusion Network for Enhanced Multimodal Driver Inattention State Recognition System**
> Minjun Kim, Gyuho Choi. *Sensors* 2025, 25(18), 5819.
> [doi:10.3390/s25185819](https://doi.org/10.3390/s25185819)

STFTransNet recognizes four driver states from in-cabin frames: normal, drowsy, yawn, and inattention. It works even when accessories such as glasses or sunglasses partially occlude the face, or when lighting changes blur the image.

- **RCN two-stream encoder**: ResNet-18 with CBAM after every stage. One stream encodes the full frame, and the other encodes the MediaPipe FaceMesh face crop.
- **Cross-attention**: the face feature queries the full-frame feature.
- **TCN**: a dilated temporal convolution network (kernel 10, dilations 2/3/4, channels 512→256→128). It runs along the sequence of frames in a mini-batch. The TCN output is added to a residual projection of the frame feature before classification.

## Installation

```bash
git clone https://github.com/minnj1025/STFTransNet.git
cd STFTransNet
pip install -r requirements.txt
```

## Data (NTHU-DDD)

The dataset is not included. Obtain NTHU-DDD (Weng et al., ACCV 2016 Workshops) from its provider.

1. Sample the videos at 10 fps.
2. Name each frame `<video>_<frame>_<fatigue>_<eye>_<head>_<mouth>.jpg`, using the four NTHU-DDD annotations.
3. Group the frames into the four states described in Section 4 of the paper.
4. Crop faces with MediaPipe FaceMesh. Frames where no face is detected have no face crop and get an all-zero face input.

```
frames/{train,test}/{normal,drowsy,yawn,inattention}/*.jpg
faces/{train,test}/{normal,drowsy,yawn,inattention}/*.jpg   (same file names)
```

`splits/nthu_ddd/{train,test}.txt` lists every image of the paper's split (71,755 training and 17,941 test frames) in the order it was loaded. The validation set is a stratified 20% of the training list, drawn with `--seed`.

## Usage

```bash
# train (validation weighted-F1 selects the checkpoint), then test
python train.py --frames_root /path/to/frames --faces_root /path/to/faces

# evaluate a checkpoint
python train.py --frames_root /path/to/frames --faces_root /path/to/faces --eval_ckpt best_model.pth
```

Defaults follow the paper's NTHU-DDD run:
- AdamW (lr 1e-4, weight decay 1e-3) with ReduceLROnPlateau.
- Batch size 64, 20 epochs, and split seed 41.
- Mixed precision.

Because the TCN runs along the batch, the batch size and the test order are part of the evaluation protocol. A short last batch is filled with repeated samples.

## Results

The paper reports the results on NTHU-DDD, StateFarm, and YawDD (Tables 7–9). This repository covers the NTHU-DDD experiment.

Evaluating the checkpoint of the paper's NTHU-DDD run with this code, on PyTorch 2.11, gives **95.79% accuracy and 0.958 weighted F1** on the test list. The paper's 95.86% was measured in the original training environment. Running the original notebook code on PyTorch 2.11 gives the same 95.79%, so the difference comes from the software environment.

## Citation

```bibtex
@article{kim2025stftransnet,
  title   = {STFTransNet: A Transformer Based Spatial Temporal Fusion Network for Enhanced Multimodal Driver Inattention State Recognition System},
  author  = {Kim, Minjun and Choi, Gyuho},
  journal = {Sensors},
  volume  = {25},
  number  = {18},
  pages   = {5819},
  year    = {2025},
  doi     = {10.3390/s25185819}
}
```

## License

The code is released under the MIT License. NTHU-DDD is governed by its own license.
