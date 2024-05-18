[![IEEE ICASSP Signal Processing Grand Challenge Top Performer(Rank 3)](https://img.shields.io/badge/IEEE%20ICASSP%20Signal%20Processing%20Grand%20Challenge%20Winner-2023-yellow.svg?style=for-the-badge)](https://2023.ieeeicassp.org/signal-processing-grand-challenges/)

# IEEE ICASSP Signal Processing Grand Challenge 2023 - e-Prevention: Person Identification and Relapse Detection from Continuous Recordings of Biosignals (Track 1 Implementation)

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Xplore-blue.svg?style=for-the-badge)](https://ieeexplore.ieee.org/abstract/document/10097005)


## About the Challenge task

Track 1 of this challenge released a corpus of longitudinal data from 46 unique users measured for a period of 2.5 months. The dataset consists of continuous measurements from a smartwatch providing an accelerometer, gyroscope, heart rate (HR), heart rate variability (HRV), and sleep status data. There is also data for the user’s physical activities and the calories
burned. We refer to the former set of features as vital signs and the latter as physical signs in this paper. The task is to identify the wearer of the watch using these raw signals. The
baseline performance on the validation set for this task was 62%. In this paper, we present a novel way of modeling behavior patterns of individuals through multi-stage fusion, and an
imputation-free technique to handle missing features. This approach provides an accuracy of 91.36% on the test set and secured third position (Team name : AI_Bezzie) in the challenge. 

![MAFN](https://github.com/payalmohapatra/ICASSP-SPGC-Track-1-2023/blob/main/ICASSP_EPrevention_GC_camera_ready_wo_inference.png)

## Codebase

### Dependencies
The required packages are listed in requirements.txt and can be installed as :
```
pip install -r requirements.txt
```

Download the preprocessed dataset and the saved model checkpoint from [here](https://drive.google.com/drive/folders/1CvT2-J_DdDtUvYWQImMMQGcmkmBl3FGS?usp=sharing).


### Training

Use the following command to train the model from scratch using the preprocessed data which are assumed to be in the folder saved_var/.

```

python training_person_id.py

```

Note : We provide the final version of the best hyperparameters for reproduction.

In the folder data_prep/ additional scripts used for preparing the data to the format in saved_var/ is provided.


### Evaluation

We provide a notebook (evaluation_tutorial_notebook.ipynb) to verify our results on the labeled validation set using the checkpoint available [here](https://drive.google.com/file/d/1444wvkD6kjUjZuhWncTyKaDUcXXO8r0X/view?usp=drive_link).



## Citation

If you use this machine learning codebase or the insights in your research work, please consider citing:
```
@inproceedings{mohapatra2023person,
  title={Person identification with wearable sensing using missing feature encoding and multi-stage modality fusion},
  author={Mohapatra, Payal and Pandey, Akash and Keten, Sinan and Chen, Wei and Zhu, Qi},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--2},
  year={2023},
  organization={IEEE}
}
```
