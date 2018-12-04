# Jupyter notebooks 
--- 

## [Tacotron - 2](https://arxiv.org/abs/1712.05884)

A tacotron model is trained. Experiment with generation of speech using this models is found on [*Tacotron Demo.ipynb*](https://github.com/hansonrobotics/Tacotron-2/blob/master/Tacotron%20Demo.ipynb) notebook.
### Requirements
**logs-Tacotron directory** - Contains trained  models 
The models can be downloaded from [this google drive folder](https://drive.google.com/file/d/1Y9a9LZ9L3-1h0hCpd11dUN-JsZtgxZ6U/view?usp=sharing).

---

 ## Graffin-lim algorithm

Contains experiments on trainable graffin-lim algorithm. 
* [Experiment](https://github.com/hansonrobotics/Tacotron-2/blob/master/Experiments%20on%20Griffin.ipynb)
* [Speech synthesis](https://github.com/hansonrobotics/Tacotron-2/blob/master/griffin_lim_synthesis_tool.ipynb)

## Waveglow 
WaveGlow: Flow based generative network for speech synthesis
WaveGlow combines insights from Glow and WaveNet in order to provide fast, efficient and high quality audio synthesis, without the need for auto-regression.WaveGlow is implemented using only a single network, trained using only a single cost function: maximizing the likelihood of the training data, which makes the training procedure simple and stable.The current PyTorch implementation produces audio samples at a rate of more than 500 kHz on an NVIDIA V100 GPU. Mean Opinion Scores show that it
delivers audio quality as good as the best publicly available WaveNet implementation.

The three neural network based models that can synthesize speech without auto-regression: Parallel WaveNet and Clarinet  These techniques can synthesize audio at more than 500kHz on a GPU. However, these models are more difficult to train and implement than the auto-regressive models.
All three require compound loss functions to improve audio quality or problems with mode collapse. In addition, Parallel WaveNet and Clarinet require two networks, a student network and teacher network. The student networks underlying both Parallel WaveNet and Clarinet use Inverse Auto-regressive Flows (IAF). Though the IAF networks can be run in parallel at inference time, the auto-regressive nature of the flow itself makes calculation of the IAF inefficient. To overcome this, these works use a teacher network to train a student network on a approximation to the true likelihood. These approaches are hard to reproduce and deploy because of the difficulty of training these models successfully to convergence.
In WaveGlow an auto-regressive flow is unnecessary for synthesizing speech. Our contribution is a flow based network capable of generating high quality speech from mel-spectrograms. We refer to this network as WaveGlow, as it combines ideas from Glow  and WaveNet. WaveGlow is simple to implement and train, using only a single network, trained using only the likelihood loss function. Despite the simplicity of the model, the PyTorch implementation synthesizes speech at more than 500kHz on an NVIDIA
V100 GPU: more than 25 times faster than real time. Mean Opinion Scores show that it delivers audio quality as good as the best publicly available WaveNet implementation trained on the same dataset.
https://github.com/NVIDIA/waveglow
The notebook is about generating speech audio using the pre-trained wave-glow model.
The process to run the generation is listed on the notebook.
Two notebooks are available for two different commits.

* [Nov 10 Commit ](https://github.com/NVIDIA/waveglow/commit/f4c04e2d968de01b22d2fb092bbbf0cec0b6586f)
    [Colab notebook](https://colab.research.google.com/drive/1rEGqQNpSofGagSKkq08ZMGVZzMPbqu8C)
* [Nov 22 Commit](https://github.com/NVIDIA/waveglow/commit/71775e4a142f54bd5b9d3f605bcb8e38f1f3d5ca)
    [Notebook](https://github.com/hansonrobotics/Tacotron-2/blob/master/waveglow-v2/WaveGlow.ipynb)
--- 
