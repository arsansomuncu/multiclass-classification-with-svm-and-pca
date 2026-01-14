SVM Classification (Part 1)

Hyperparameter Tuning: We used GridSearchCV.

{C}: Controls the trade-off between a smooth decision boundary and classifying training points correctly. A high C aims for perfect classification (risk of overfitting), while a low C allows for a softer margin.

Gamma ($\gamma$): Defines how far the influence of a single training example reaches.Strategy: The decision_function_shape='ovr' handles the multiclass nature (One-vs-Rest), which is standard for SVC.



PCA & Comparison (Part 2)

Variance Retention: By plotting the cumulative explained variance, you will see a curve that rises steeply. We cut this off at 95% variance.

Impact:

Accuracy: Often drops slightly after PCA because some information is lost (the 5% variance we discarded). However, if the removed features were noise, accuracy might remain stable or even improve.

Computational Cost: Training time should decrease significantly. The original dataset has 561 features. PCA often reduces this to fewer than 100 components for HAR data, speeding up the SVM solver drastically.











