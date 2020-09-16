# Augmented Experiments in Material Engineering Using Machine Learning

Code accompanying the paper

> [Augmented Experiments in Material Engineering Using Machine Learning](/docs/5266.pdf)


The synthesis of materials using the principle of thermogravimetric analysis to discover new anticorrosive paints requires several costly experiments. This paper presents an approach combining empirical data and domain analytical models to reduce the number of real experiments required to obtain the desired synthesis.
The main idea is to predict the behavior of the synthesis of two materials with well-defined mass proportions as a function of temperature. As no exact equational model exists to predict the new material, we integrate a machine learning approach circumscribed by existing domain analytical models such as heating equation in order to derive a generative model of augmented experiments.


## Requirements

```
TensorFlow == 1.14.0
```

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

The dataset used in this work can be found [here](/data/).

It consists of thermal analysis of raw materials. These were collected with an SDT-Q600 version 20.9 build 20 industrial instrument that monitors the calcination of the mixtures continuously.
The instrument encompasses a pair of thermocouples within the ceramic beams that provides direct sample, reference, and differential temperature measurements from ambient to 1500 °C.
Specifically, 6 signals are monitored by the instrument, namely, temperature (°C), weight (mg), heat flow (mW), temperature difference (◦C), temperature difference (μV ), and sample purge flow (mL/min).
The dynamics of the Nitrogen gas, which constitutes the ambient atmosphere around the mixture, is set to 100 ml/min. The acquisition of the various signals was carried at a sampling rate of 2 Hz which is sensitive enough, in these kinds of applications, for capturing temperature and mass trends that may indicate regime changes.
In total, 3000 measurement points were obtained for each set of experiments. In addition to the theoretical curves of the red pigment (pig) and the calamine oxide (cala) that were obtained separately, we perform calcination of mixtures with various percentages, pi ∈ {5, 10, 15, 20, 25, 35}, of additional calamine oxide to the red pigment.

Here are some visualizations.

<p align="center">
    <img src="/img/dsc-tga-calamine.png" width="300px"/>
    <img src="/img/dsc-tga-pigment.png" width="300px"/>
</p>

<p align="center">
    <img src="/img/tga_surface_heat-flow.png" width="22%"/>
    <img src="/img/tga_surface_temperature_difference_celsius.png" width="22%"/>
    <img src="/img/tga_surface_temperature_difference_micro.png" width="20.5%"/>
    <img src="/img/tga_surface_weight.png" width="22%"/>
</p>
<p align="center">
Figure: Simultaneous thermal and mass loss analysis of (top) calamine oxide and red pigment. (bottom) binary mixture of red pigment and additional calamine percentages. The effect of the temperature augmentation on the behavior of the red pigment is shown via weight, derivative weight, temperature difference, and heat flow curves.
Further analysis of mass loss, variation of the dissociation reaction enthalpy, and the formation of new phases can be found in Section Thermal Analysis.
</p>


## Training

To train the model(s) in the paper, run this command:

```train
```


## Evaluation

To evaluate my model, run:

```eval
```

## Results

Our model achieves the following performance :

<p align="center">
    <img src="/img/weight_reconstructions.png" width="30%">
    <img src="/img/temperature_reconstructions.png" width="30%">
</p>
<p align="center">
Figure: Obtained state space reconstructions for (left) weight and (right) temperature. We report reconstructions averaged over all evaluation setups and their corresponding perplexity.
As references, we also report the reconstructions obtained (under the same evaluation setups) using the baseline.
</p>


<p align="center">
    <img src="/img/training-loss-sgd-vs-cd.png" width="60%">
</p>
<p align="center">
    <img src="/img/sgd-vs-cg-distance-between-experiments-inside.png" width="30%">
    <img src="/img/sgd-vs-cg-distance-between-experiments-outside.png" width="30%">
</p>
<p align="center">
Figure: Comparing the performances of SGD vs. CG: (top) evolution of the training loss as a function of the number of training epochs.
(bottom) Extent of the reconstructions as a function of the distance from the set of training to the set of validation experiments (Inside and outside circumscribed regions of the state space, respectively).
We repeat the evaluation for 10 times with different random seeds and report the median and the best validation performance of the models.
</p>


<p align="center">
    <img src="/img/evolution-of-weight-reconstuction-error.png" width="31.5%">
    <img src="/img/evolution-of-temperature-reconstuction-error.png" width="30%">
</p>
<p align="center">
Figure: Reconstruction performances at specific percentages of additional calamine oxide. We compare the reconstructions, of (left) weight and (right) temperature, obtained using the baseline vs. the regularized models.
Results averaged over all possible distances to the set of training experiments.
</p>
