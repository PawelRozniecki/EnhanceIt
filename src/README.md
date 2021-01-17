# Pytorch implementation of GAN based Video Super Resolution 

The goal of this project is to Upscale any video to a higher resulution using a Generative Adversial Network. 


## REQUIREMENTS 
- Python 3.7+
- PyTorch
- OpenCV 
- FFMPEG
- Prefarably NVIDIA CUDA enabled GPU 

```
pip3 install torch torchvision torchaudio
pip3 install ffmpeg
pip3 install opencv-python
```


## Datasets

For the training I've used DIV2K dataset that can be found here: [Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

## Papers 
[Inspired by  Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network Paper](https://arxiv.org/abs/1609.04802)


## HOW TO 

In order to run video enhancement run **App.py**
It takes 3 arguments:

-  --path: path to a video you want to enhance
-  --scale_factor: An upscaling factor. Choice between 2 and 4. Default one is 4 
-  --arcnn: Bool Check to true, to remove compression artifacts from a video first 

```

python3 App.py --path --scale_factor --arcnn

```

#### Example 

```
python3 App.py --path movie.mp4 --scale_factor 2 --arcnn False
```

#### Steps

1. Extracts all frames and saves them to *extracted_frames* folder 
2. If --arcnn argument was set to True then compression artifacts will be removed from the video and frames will be saved in *arcnn_frames* folder, otherwise it skips this step
3. Upscales frames using SRGAN from extracted_frames folder, and saves them in *enhanced_frames* folder
4. Upscales frames using *Bicubic interpolation* and saves them in bicubic_resampling folder
5. Combines all of  the frames from  *enhanced_frames and Bicubic interpolation folders* into a video. Bicubic and SRGAN enhanced videos are saved as **out.mp4** in those folders 
