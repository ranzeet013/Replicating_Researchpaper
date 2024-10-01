# A Comparison Study: Support Vector Machines for Binary Classification in Machine Learning 

Paper Link : [A Comparison Study: Support Vector Machines for Binary Classification in Machine Learning](https://sci-hub.se/10.1109/BMEI.2011.6098517)

This project replicates and extends the comparison study of various Support Vector Machine (SVM) algorithms for binary classification as described in the paper. The primary focus is on evaluating several SVM variants in both **supervised** and **semi-supervised learning frameworks**, including **LapSVM**.

## Datasets Used

The following datasets were used for the evaluation:

1. **Colon-cancer dataset**: A dataset with 62 examples and 2000 features, often used in biomedical research for binary classification tasks.
2. **Breast-cancer dataset**: A widely-used dataset with 683 examples and 10 features, sourced from the UCI Machine Learning Repository.
3. **a1a dataset**: Contains 1605 examples with 123 features, often used for benchmarking binary classification algorithms.
4. **svmguide1 dataset**: Contains 3089 examples and 4 features, specifically used for SVM benchmarking.

## SVM Variants Tested

1. **Linear SVM**: Utilizes a linear kernel to classify data.
2. **Polynomial SVM**: Uses a polynomial kernel for classification.
3. **RBF SVM (Radial Basis Function)**: A popular kernel used for non-linear data classification.
4. **LapSVM (Laplacian SVM)**: A semi-supervised learning algorithm that utilizes both labeled and unlabeled data to enhance generalization performance.

## Results

The accuracy of each SVM algorithm on the various datasets is shown below:

### Colon-cancer Dataset

| Algorithm         | Accuracy |
|-------------------|----------|
| Linear SVM        | 0.6842   |
| Polynomial SVM    | 0.4211   |
| RBF SVM           | 0.4211   |

### Breast-cancer Dataset

| Algorithm         | Accuracy |
|-------------------|----------|
| Linear SVM        | 0.9649   |
| Polynomial SVM    | 0.6316   |
| RBF SVM           | 0.9415   |

### a1a Dataset

| Algorithm         | Accuracy |
|-------------------|----------|
| Linear SVM        | 0.8838   |
| Polynomial SVM    | 0.4793   |
| RBF SVM           | 0.8631   |

### svmguide1 Dataset

| Algorithm         | Accuracy |
|-------------------|----------|
| Linear SVM        | 0.8770   |
| Polynomial SVM    | 0.9234   |
| RBF SVM           | 0.8447   |

### LapSVM (Semi-supervised Learning)

| Dataset            | LapSVM Accuracy |
|--------------------|------------------|
| Combined (Breast-cancer) | 0.9279       |

## Observations

- **Colon-cancer dataset**: Linear SVM performed the best with an accuracy of 0.6842, while both Polynomial and RBF SVMs underperformed.
- **Breast-cancer dataset**: Linear SVM achieved a near-perfect accuracy of 0.9649. The RBF SVM also performed well, but the Polynomial SVM lagged significantly.
- **a1a dataset**: Linear SVM performed best with an accuracy of 0.8838, while RBF SVM followed closely. Polynomial SVM was again the weakest performer.
- **svmguide1 dataset**: Polynomial SVM performed the best, surpassing both Linear and RBF SVMs.
- **LapSVM**: The semi-supervised LapSVM achieved an accuracy of 0.9279, showing improved performance when leveraging both labeled and unlabeled data.

## Conclusion

From the experiments:
- **Linear SVM** performed well across most datasets, especially when data is relatively clean and linearly separable.
- **Polynomial SVM** consistently underperformed compared to other variants, suggesting that higher-order feature interactions may not be beneficial for these datasets.
- **RBF SVM** performed well on some datasets but was outperformed by other models in certain cases.
- **LapSVM** demonstrated strong performance by incorporating unlabeled data, suggesting that it is a promising approach for semi-supervised learning, especially in cases where labeled data is sparse.

Further work can explore the behavior of these algorithms on larger datasets and investigate ways to handle outliers for improved classifier performance.



