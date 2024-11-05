# Scaling Effect

We delve deeper into analyzing the model's layer depth and scalability, employing the CricketX dataset for rigorous testing.

## Effect of GTF layers on results

The result is shown below. The model performs optimally when configured with 3 layers of the GTF block. Nonetheless, an intriguing observation emerges as the ratio of labels increases; in this scenario, a GTF configuration with 5 layers proves most effective. We postulate that this shift stems from the augmented volume of label data, necessitating a more expansive network infrastructure to adequately capture and learn the intricate mappings between the labels and the underlying data.

| Label ratio | 1     | 2     | 3         | 4     | 5         |
| ----------- | ----- | ----- | --------- | ----- | --------- |
| 0.1         | 61.84 | 62.28 | **62.89** | 59.59 | 58.04     |
| 0.2         | 70.59 | 69.66 | **72.49** | 71.14 | 70.38     |
| 0.4         | 79.54 | 80.10 | 79.44     | 82.71 | **83.67** |
| 1.0         | 89.73 | 87.03 | 90.94     | 89.97 | **91.45** |
| Avg         | 75.43 | 74.77 | **76.44** | 75.85 | 75.89     |



## Effect of kernel sizes on results

The result is shown below. We investigate a range of kernel size configurations, including [5,9,19,39], [9,19,39,59], [19, 39,59,109], and [5,19,59,109].  Our findings highlight that the configuration [5, 19, 59, 109] yields the optimal performance. We attribute this superiority to two key factors. Firstly, the employment of larger convolutional kernels plays a pivotal role in enhancing overall performance, suggesting that a broader receptive field benefits feature extraction. Secondly, we postulate that the selection of kernels with substantial parameter disparities fosters orthogonality among the extracted features, thereby enriching the feature space and improving the model's discriminatory capacity. This strategic combination of kernel sizes appears to optimize feature representation and boost the model's performance.

| label ratio | [5,9,19,39] | [9,19,39,59] | [19,39,59,109] | [5,19,59,109] |
| ----------- | ----------- | ------------ | -------------- | ------------- |
| 0.1         | 59.69       | 56.56        | 56.25          | **62.89**     |
| 0.2         | 71.43       | **73.50**    | 69.31          | 72.49         |
| 0.4         | 77.64       | 78.60        | 75.88          | **79.44**     |
| 1.0         | 89.62       | 87.85        | **91.24**      | 90.94         |
| Avg         | 74.60       | 74.13        | 73.17          | **76.44**     |



