**Why can SOP** **contribute performance?** 

 We first theoretically analyze the effectiveness of SOP. According to Theorem 3 in the manuscript, the uniform stability is effected by the positive constants of pretext task head   . However,   is a theoretical value that cannot be directly computed. We approximate   using the Frobenius norm of the unsupervised classification head network, i.e.,   . We additionally report the average of approximate positive constants   on the different label ratios 10%, 20%, 40%, and 100%, as shown in Table 1. The results show that the   values of SOP task are consistently surpass Temporal prediction and Forecasting tasks on the six real-world datasets. We can confirm that SOP task significantly increases the   value, as evidenced by a t-test result with   .

Table 1 The results report the approximate positive constants   for the label ratio of 10%, 20%, 40% and 100% on six datasets.

| Methods  | Datasets            |           |           |           |           |           |           |      |
| -------- | ------------------- | --------- | --------- | --------- | --------- | --------- | --------- | ---- |
| Backbone | Pretext task        | CricketX  | Insect    | UWave     | Epileptic | XJTU      | MFPT      |      |
| GTF      | Temporal prediction | 4.53      | 4.01      | 7.39      | 3.36      | 5.20      | 5.65      |      |
| GTF      | Forecasting         | 22.62     | 10.22     | 19.67     | 18.80     | 21.35     | 23.64     |      |
| GTF      | SOP                 | **23.72** | **12.31** | **21.11** | **31.36** | **44.66** | **45.00** |      |
|          |                     |           |           |           |           |           |           |      |

Experimentally, we conduct a series of ablation studies to analyze the effects of different pretexts, as shown in Table 2. The results show that the proposed SOP task has a better performance compared with the Temporal prediction task. The order relation achieves an improvement of 7.53% in terms of the average accuracy on the six datasets. Additionally, we observe that the SOP task yields a 11.74% improvement compared to the scenario without pretext task training. 

Table 2 The ablation studies on the pretext task report the average results for the label ratio of 10%, 20%, 40% and 100% on the six datasets.



| Methods  | Datasets             | Avg       |           |           |           |           |           |           |        |
| -------- | -------------------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ------ |
| Backbone | Pretext task         | CricketX  | Insect    | UWave     | Epileptic | XJTU      | MFPT      |           |        |
| GTF      | -                    | 55.60     | 58.41     | 77.59     | 77.25     | 90.34     | 79.22     | 73.07     | +11.74 |
| GTF      | Forecasting          | 68.20     | 53.44     | 88.06     | 75.63     | **97.99** | 90.77     | 79.02     | +5.79  |
| GTF      | Temporal  prediction | 67.15     | 49.21     | 90.74     | 79.79     | 87.27     | 89.69     | 77.28     | +7.53  |
| GTF      | SOP                  | **76.44** | **67.24** | **93.93** | **80.35** | 96.40     | **94.51** | **84.81** | -      |



**Why can GTF** **contribute performance?** 

The core component of GTF block is the Long to Short Module (LSM). Empirically, this module mixes trends with the primary goal of obtaining more diverse multi-trend features. It progressively aggregates detailed information from fine to coarse scales, integrating micro-trend information with macro-trend information, ultimately achieving a multi-trend mixture.

Experimentally, we evidence the effectiveness of the GTF block by comparing it to classic and state-of-the-art backbones, as shown in Table 3. The experimental results reveal notable performance improvements for the GTF, with enhancements of 13.21%, 6.45%, 5.88%, and 10.97% compared to ResNet, ITime, Transformer, and Mamba, respectively. Particularly noteworthy is the 6.45% performance boost over ITime, indicating significant benefits from generating fine-grained multi-trend features. Additionally, we supplement the GTF block with bottom-up and top-down mixing strategies. The results show that short-term trend features guided by long-term trend features are more effective than long-term trend features guided by short-term trend features.

Table 3 The ablation studies on the backbone report the average results for the label ratio of 10%, 20%, 40% and 100% on six datasets.   indicates bottom-up mixing while   indicates top-down.

| Methods     | Datasets     | Avg       |           |           |           |           |           |           |        |
| ----------- | ------------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ------ |
| Backbone    | Pretext task | CricketX  | Insect    | UWave     | Epileptic | XJTU      | MFPT      |           |        |
| Resnet      | SOP          | 57.32     | 31.46     | 71.26     | **80.99** | 94.47     | 94.13     | 71.60     | +13.21 |
| ITime       | SOP          | 59.90     | 59.40     | 82.29     | 80.07     | 96.13     | 92.39     | 78.36     | +6.45  |
| Transformer | SOP          | 65.04     | 56.99     | 89.62     | 79.65     | 93.55     | 88.71     | 78.93     | +5.88  |
| Mamba       | SOP          | 67.07     | 41.18     | 91.70     | 80.04     | 89.13     | 73.94     | 73.84     | +10.97 |
| GTF         | SOP          | 69.96     | 50.00     | 90.48     | 76.55     | 77.99     | 92.34     | 76.22     | +8.59  |
| GTF         | SOP          | **76.44** | **67.24** | **93.93** | 80.35     | **96.40** | **94.51** | **84.81** | -      |

**Why can TOFL** **contribute performance?**  

Finally, we perform ablation studies on the SOP task and GTF block to validate the effectiveness of TOFL, as shown in Table 4. By modifying the model structure, we find that the SOP alone yields a 7.53% improvement, the GTF block alone results in a 6.45% improvement, and the integration of both structures leads to a 7.83% improvement. These results indicate that the multi-trend features and order features have a substantial influence in the time series classification.

Table 4 The ablation studies on the proposed methods report the average results for the label ratio of 10%, 20%, 40% and 100% on six datasets.

| Methods | Datasets | Avg       |           |           |           |           |           |           |       |
| ------- | -------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | ----- |
| GTF     | SOP      | CricketX  | Insect    | UWave     | Epileptic | XJTU      | MFPT      |           |       |
|         |          | 65.43     | 48.43     | 91.91     | **80.81** | 86.92     | 88.36     | 76.98     | +7.83 |
|         |          | 59.90     | 59.40     | 82.29     | 80.07     | 96.13     | 92.39     | 78.36     | +6.45 |
|         |          | 67.15     | 49.21     | 90.74     | 79.79     | 87.27     | 89.69     | 77.28     | +7.53 |
| GTF     | SOP      | **76.44** | **67.24** | **93.93** | 80.35     | **96.40** | **94.51** | **84.81** | -     |