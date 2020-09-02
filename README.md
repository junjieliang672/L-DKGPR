# L-DKGPR
 Longitudinal Deep Kernel Gaussian Process



## Notes

1. The raw data: we do not include the raw data file since some of them are only available after application. We instead leave the official website for downloading the raw data in the corresponding preprocessing file.
2. The code for data preprocessing are included in the folder `preprocess`
3. The data after preprocessing are included in the folder `data`



## Testing L-DKGPR

Code for our model is in file `L-DKGPR`. To run our code, you should first format your data and save it using `scipy.io.savemat(dataset)` where `dataset` is the target data file, which contains the following fields:

* trainX: The covariate matrix for the training data. A 2D numpy array with size $N_{train}\times P$. 
* trainY: The training labels. A 1D numpy array with size $N_{train}$
* testX: The covariate matrix for the test data. A 2D numpy array with size $N_{test}\times P$
* testY: The testing labels. A 1D numpy array with size $N_{test}$
* trainId: The individual id corresponding to each row of the training data.
* testId: The individual id corresponding to each row of the test data.
* trainOid: The observation id (timestamp) corresponding to each row of the training data.
* testOid: The observation id (timestamp) corresponding to each row of the test data.

To run our algorithm, you should first try`python L-DKGPR.py --help` to check the available parameters. Basically, you can just use the default setting or change the `--file` to locate your target data file.



## Baseline models

The code for our baseline models are located in baseline_model. 



### Citation

```
@article{liang2020longitudinal,
  title={Longitudinal Deep Kernel Gaussian Process Regression},
  author={Liang, Junjie and Wu, Yanting and Xu, Dongkuan and Honavar, Vasant},
  journal={arXiv preprint arXiv:2005.11770},
  year={2020}
}
```

