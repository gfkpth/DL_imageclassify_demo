# Image classification demo

This project demos image classification with convolutional neural networks using the [Animals-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data).

The dataset contains images of animals categorised for 10 different labels. The category sizes and image sizes are not uniform, so some pre-processing is required.

The best performing model (based on [EfficientNetV2S](https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function); see [also here](https://arxiv.org/abs/2104.00298)) reached an accuracy of `0.9767`. The runner-up, based on [MobileNet-V2](https://keras.io/api/applications/mobilenet/#mobilenetv2-function) (see [also here](https://arxiv.org/abs/1801.04381)), has only a slightly lower accuracy at `0.965623`, but is much smaller and presumably more efficient.

For comparison, below the model sizes:

**effnet2S_small_finetune**

- Test set accuracy: 0.9767
- Total params: 25,211,268 (96.17 MB)
- Trainable params: 2,274,698 (8.68 MB)
- Non-trainable params: 18,387,168 (70.14 MB)
- Optimizer params: 4,549,402 (17.35 MB)

**mobnet2_dense256_finetune**

- Test set accuracy: 0.965623 
- Total params: 3,249,508 (12.40 MB)
- Trainable params: 330,506 (1.26 MB)
- Non-trainable params: 2,257,984 (8.61 MB)
- Optimizer params: 661,018 (2.52 MB)

# Structure

- [01-preprocessing.ipynb](01-preprocessing.ipynb): 
  - creation of a csv overview of the raw image files (<data/overview.csv>)
  - rescaling and reformating of images to 224x224x3 RGB
  - creation of csv for cleaned data (<data/overview.csv>)
- [02-CNN-modeltraining.ipynb](02-CNN-modeltraining.ipynb): contains a variety of handbuilt CNNs
- [03-Transfer Learning.ipynb]('03-Transfer Learning.ipynb'): contains the transfer learning models and some summary and additional mini-test at the end 
- <auxiliary.py>: various auxiliary functions for notebooks 02 and 03 (encapsulation is not ideal, contains some global variables that need to be kept consistent with the notebooks)
- <results.csv>: table of training and test results
- <results_200_200.csv>: 
- <data/>
  - raw images are expected in a subfolder "raw-img" (unpack from the kaggle dataset)
  - processed images will be generated in subfolder "processed"
  - csv files for data overview, cleaned dataset and train, validation and test subsets
- <models/>
  - contains some of the models trained (a choice of the best results)


# Result overview

|    | model_id                           |   learning_rate |   epochs |   acc_train |   accuracy |
|---:|:-----------------------------------|----------------:|---------:|------------:|-----------:|
| 30 | effnet2S_small_finetune            |          1e-05  |       35 |    0.988264 |   0.9767   |
| 29 | effnet2S_small                     |          0.0005 |       12 |    0.974127 |   0.97479  |
| 33 | mobnet2_dense256_finetune          |          1e-05  |       30 |    0.966921 |   0.965623 |
| 31 | mobnet2_dense256                   |          0.0005 |       13 |    0.957696 |   0.961421 |
| 25 | mobnet2_small_tune20_15epochs      |          0.0001 |       10 |    0.965884 |   0.960275 |
| 28 | resnet2_small_finetune             |          1e-05  |       30 |    0.993832 |   0.959129 |
| 24 | mobnet2_small_tune20               |          0.0001 |        5 |    0.960426 |   0.957601 |
| 23 | mobnet2_small_tune20               |          1e-05  |        5 |    0.953603 |   0.956837 |
| 32 | mobnet2_dense256                   |          0.0005 |       13 |    0.961408 |   0.956073 |
| 19 | mobnet2_2dense                     |          0.0005 |       15 |    0.941103 |   0.954927 |
| 17 | mobnet2_small                      |          0.0007 |       15 |    0.956277 |   0.953018 |
| 18 | mobnet2_3dense                     |          0.0005 |       15 |    0.913319 |   0.951872 |
| 27 | resnet2_small_finetune             |          0.0001 |       30 |    0.989574 |   0.94958  |
| 26 | resnet2_small                      |          0.0007 |       15 |    0.954476 |   0.94576  |
| 39 | conv4invto128_dense256_upsamp      |          0.0005 |       80 |    0.875676 |   0.719633 |
| 20 | conv5invto128_dense1_80total       |          0.0005 |       10 |    0.824945 |   0.713904 |
| 15 | conv5invto128_dense1_35extraepochs |          0.0005 |       70 |    0.815284 |   0.712758 |
| 21 | conv4invto128_dense1               |          0.0005 |       80 |    0.829913 |   0.710848 |
| 37 | conv4invto128_dense256             |          0.0005 |       80 |    0.836572 |   0.702827 |
| 41 | conv5invto128_dense256_upsamp      |          0.0002 |       80 |    0.866392 |   0.700917 |
| 38 | conv4invto128_pool_dense256_upsamp |          0.0005 |       80 |    0.831169 |   0.683728 |
| 13 | conv4invto128_dense1               |          0.0005 |       35 |    0.787009 |   0.678762 |
| 40 | conv5invto128_dense256             |          0.0002 |       80 |    0.816539 |   0.670741 |
|  8 | conv4lesspoll_dense1               |          0.0005 |       35 |    0.740393 |   0.668831 |
| 10 | conv3_dense1                       |          0.0005 |       35 |    0.709116 |   0.663484 |
| 35 | conv4inv8to128_pool_dense196       |          0.0001 |       80 |    0.783297 |   0.658136 |
| 14 | conv5invto128_dense1               |          0.0005 |       35 |    0.723799 |   0.653552 |
| 11 | conv3invert_dense1                 |          0.0005 |       35 |    0.716758 |   0.650497 |
|  4 | conv4small_dense1                  |          0.0007 |       30 |    0.687882 |   0.646295 |
| 22 | conv4lesspoll_dense1               |          0.0005 |       80 |    0.679803 |   0.645913 |
| 36 | conv4inv8to128_dense96             |          0.0001 |       80 |    0.728603 |   0.645531 |
|  6 | conv5small_dense1                  |          0.0007 |       30 |    0.677566 |   0.642857 |
| 34 | conv4inv128_pool_dense256          |          0.0001 |       80 |    0.757042 |   0.639037 |
|  9 | conv3_dense2                       |          0.0005 |       35 |    0.613537 |   0.617647 |
|  3 | conv4dense2                        |          0.0007 |       25 |    0.647926 |   0.614591 |
|  7 | conv4small_dense2_lowlr            |          0.0005 |       30 |    0.648144 |   0.611154 |
| 12 | conv4lesspollinv_dense1            |          0.0005 |       35 |    0.673253 |   0.608098 |
|  5 | conv5_dense1                       |          0.0007 |       30 |    0.599782 |   0.585943 |
|  1 | base_upsamp_augment+               |          0.0007 |       15 |    0.591745 |   0.566845 |
|  2 | base_meansamp_augment+             |          0.0007 |       15 |    0.581332 |   0.542399 |
|  0 | base_downsamp_augment+             |          0.0007 |       15 |    0.510178 |   0.525974 |
| 16 | mobnet2_small_notraining           |          0.0007 |       15 |  nan        |   0.07945  |

