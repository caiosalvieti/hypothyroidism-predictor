## Principal Component Analysis (PCA)

### Overview:
Principal Component Analysis (PCA) is an unsupervised statistical technique used to reduce data dimensionality while retaining the maximum amount of variability. Its interpretation requires a critical approach, especially when analyzing principal components and their corresponding loadings, which indicate the weight of each variable within each component.

According to Malinowski’s criteria, variables with loadings greater than or equal to 0.7 are considered significant.

### Graphical Analysis

Graph 1: TSH vs. TT4 with Age
The relationship between TSH and TT4 demonstrates distinct patterns: high TT4 levels are typically associated with low TSH levels, and vice versa. Some patients exhibit balanced hormone levels. Age introduces an additional dimension, though no clear pattern is observed.

Graph 2: TSH vs. T4U with Age
High T4U values do not consistently correspond with changes in TSH levels. Certain cases show TSH levels ranging from 100 to 500 with T4U values between 1.0 and 1.5.

Graph 3: T3 vs. FTI with Age
There is a strong correlation between T3 and FTI. Most patients cluster around FTI ≈ 300 and T3 ≈ 6, with increases in one often mirrored by increases in the other.

### Component Variance

Component variance estimation is a statistical method that measures the variability within a system. This technique provides valuable insights when analyzing randomly selected items across different topics.

In the plotted data, a significant difference is observed between PC1 and PC6. This suggests that variations are most informative up to PC3, effectively highlighting specific component measures.

## PCA Analysis

PCA Summary - Part 1:
PC1: Significant variables are T3 (0.789) and TT4 (0.920). Positive PC1 values indicate high T3 and TT4 levels, while negative values suggest low levels of these hormones.
PC2: The most significant variable is T4U (0.863).
Patients with hypothyroidism typically exhibit higher positive T4U levels, while those without the condition often have elevated T3 and TT4 levels. The differences between these groups are substantial, and these parameters could be meaningful when considered alongside other variables and correlations.

PCA Summary - Part 2:
PC1: Again highlights strong correlations with T3 (0.789) and TT4 (0.920).
PC3: Age is the most significant variable (0.945).
This plot primarily demonstrates the correlation between Age, T3, and TT4. Higher levels of T3 and TT4 appear to be associated with patient age. Notably, patients with hypothyroidism show diverse age ranges and varying T3 and TT4 levels.

PCA Summary - Part 3:
PC1: Significant variables remain T3 (0.789) and TT4 (0.920).
PC4: The significant variable is TSH (0.768).
Patients with hypothyroidism (blue) usually have a much higher concentration at the left side and the superior part of the graphic. Also, negative (green) patients are more divided from the center to the right side at the plot, which means T3/TT4 much higher is equivalent to PC1 positive and lower TSH, whereas hypothyroidism is usually associated with higher TSH or PC4 positive and lower T3/TT4 or PC1 negative.

PCA Summary - Part 4:
PC2: The most significant variable is T4U (0.863).
PC3: Age is the most significant variable (0.945).
The plot is mainly based on how PC3 shows that the factor age isn’t a huge topic to measure a huge impact at the variations, whatever the distribution between the patients mostly with hypothyroidism are more at the right side at the plot, which makes sense because hypothyroidism has higher T4U, between that the PC3 has higher concentration of patients between -2 and +1, with some concentration of negative values which indicates the relation with youth and medium age patients. The significant number here is the elevation of T4U and the hypothyroidism stays between PC2 positive and under zero PC3.

PCA Summary - Part 5:
PC2: The most significant variable is T4U (0.863).
PC4: The significant variable is TSH (0.768).
The plot shows how the patients with hypothyroidism are more concentrated at the higher values of PC4, which means also a higher TSH, as PC2 values as well which go at the medium values to high values. Between the negative patients is visible a bigger concentration at the bottom part of the graph which shows that PC4 lower is equivalent to regular TSH, horizontally divided at PC2, but without huge climbs.

PCA Summary - Part 6:
PC3: Age is the most significant variable (0.945).
PC4: The significant variable is TSH (0.768).
This plot primarily demonstrates the correlation between Age and TSH. Higher levels of TSH appear to be associated with patient age. Notably, patients with hypothyroidism show diverse age ranges and varying TSH levels.

## K - Nearest Neighbors

K-Nearest Neighbors (KNN) is a classification algorithm that operates by grouping similar data points located near each other based on proximity.

Method: The KNN algorithm finds the K nearest neighbors within the selected groups using a distance metric, defined by K.
Characteristics: KNN is easy to implement, adaptive, and simple but does not scale very well with larger data sets, leading to a slow process and a significant curse of dimensionality.
	Next steps numbers Y

### Accuracy: 0.9851116625310173 
- This number represents the overall accuracy of the model, indicating that 98.51116625310173% of the predictions were correct.

### Confusion Matrix:

### [[379, 0],
 ### [6, 18]]

The confusion matrix is a 2x2 table that shows the performance of a classification model. In this case, the first row represents the true positives (379) and false negatives (0), while the second row represents the false positives (6) and true negatives (18).

### Precision: 0.985343688569495
- Precision is calculated as the number of true positives divided by the sum of true positives and false positives. In this context, it means that 98.5343688569495% of the predicted positives were actually positive.

### Recall: 0.9851116625310173 
- Recall is calculated as the number of true positives divided by the sum of true positives and false negatives. This indicates that 98.51116625310173% of the actual positives were correctly identified by the model.

### F1 Score: 0.9841066719127857 
- The F1 score is the harmonic mean of precision and recall, providing a balance between the two metrics. It suggests that the model has a high level of both precision and recall.

### MCC: 0.859250655112749 
- Matthews Correlation Coefficient (MCC) is a measure of the quality of binary classification, considering all four categories of the confusion matrix. A value of 0.859250655112749 indicates a strong correlation between the predicted and actual labels.

UNDERSAMPLING AND OVERSAMPLING

. . .

### Conclusion


