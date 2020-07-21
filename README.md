# NeuralSNR

**Gravitational-wave signal-to-noise interpolation via neural networks**

> Computing signal-to-noise ratios (SNRs) is one of the most common tasks in gravitational-wave data analysis.  While a single SNR evaluation is generally fast, computing SNRs for an entire population of merger events could be time consuming. We compute SNRs for aligned-spin binary black-hole mergers as a function of the (detector-frame) total mass, mass ratio and spin magnitudes using selected waveform models and detector noise curves, then we interpolate the SNRs in this four-dimensional parameter space with a simple neural network (a multilayer perceptron). The trained network can evaluate 10<sup>6</sup> SNRs on a 4-core CPU within a minute with a median fractionalerror below 10<sup>−3</sup>. This corresponds to average speed-ups by factors in the range [120,7.5×10<sup>4</sup>], depending on the underlying waveform model. Our trained network (and source code) is publicly available online, and it can be easily adapted to similar multidimensional interpolation problems.

This repo contains the network trained in arXiv:XXXX.XXXXX. If you make use our code or pre-trained, please consider citing us.

The code is developed and maintained by [Kaze Wong](https://github.com/kazewong). Please open an issue on GitHub if you want to report bugs or make suggestions. For any other problem, feel free to contact me with [kazewong@jhu.edu](kazewong@jhu.edu)

The pytorch template used for training is from [PyTorch Project Template](https://github.com/moemen95/PyTorch-Project-Template)

# Prerequiste

This code depends on the following packages:

`numpy, astropy, scipy, pytorch, h5py`

Please also find relevant packages used in the pytorch project template repo.

# Using pretrained model

**Disclaimer: we encourage the use of our network only for prototyping purposes. For any publication-level calculation, we strongly recommend users to train their own network. The performance and accuracy of a neural network crictically depends on the training data, and there is no single set of assumptions fitting in any analysis. The goal of our paper is to encourage the community to use neural networks in GW applications, not to answer particular scientific questions. SO TRAIN YOUR OWN NETWORK. You have been warned :).**



# Training your own network
