# A Remark on Concept Drift for Dependent Data

Experimental code of conference paper. [Paper](TODO) [(ArXive version) TODO](https://arxiv.org/abs/)

## Abstract

Concept drift, i.e., the change of the data generating distribution, can render machine learning models inaccurate. Several works address the phenomenon of concept drift in the streaming context usually assuming that consecutive data points are independent of each other. To generalize to dependent data, many authors link the notion of concept drift to time series. In this work, we show that the temporal dependencies are strongly influencing the sampling process. Thus, the used definitions need major modifications. In particular, we show that the notion of stationarity is not suited for this setup and discuss alternatives. We demonstrate that these alternative formal notions describe the observable learning behavior in numerical experiments.

**Keywords:** Concept Drift, Dependent Data, Concept Drift Detection

## Requirements

* Python 
* Numpy, SciPy, Pandas, Matplotlib
* scikit-learn
* chpt (Kernel Drift Detector)
* statsmodels

## Usage

To run the experiments, there are three stages 1. create the datasets (`--make`) which creates the datasets and stores them in a local directory, 2. splits the experimental setups in several chunks (`--setup #n`) for parallel processing on different devices, and 3. running the experiments (`--run_exp #n`) which runs the chunk as indicated by the command line attribute.

## Cite

Cite our Paper
```
TODO
```

## License

This code has a MIT license.
