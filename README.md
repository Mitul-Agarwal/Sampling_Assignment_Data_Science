# Credit Card Fraud Detection with Sampling Techniques and Machine Learning Models

This project applies various sampling techniques to balance an imbalanced dataset of credit card transactions. Several machine learning models are trained and evaluated on the sampled data to determine the best-performing combination of sampling techniques and models.

## Dataset

The dataset used for this project contains credit card transaction data, where the `Class` column indicates whether a transaction is fraudulent (`1`) or legitimate (`0`). The dataset is highly imbalanced, requiring techniques like SMOTE for balancing.

### Data Preprocessing
- Features (`X`) are separated from the target column (`Class`).
- Synthetic Minority Oversampling Technique (SMOTE) is applied to balance the class distribution.

## Sampling Techniques

Five different sampling techniques are applied to create subsets of the balanced dataset:
1. **Technique1**: Random sample with a specific random state.
2. **Technique2**: Random sample with a different random state.
3. **Technique3**: Systematic sampling (based on intervals).
4. **Technique4**: Another random sample with a different random state.
5. **Technique5**: Yet another random sample with a different random state.

## Machine Learning Models

The following machine learning models are used to evaluate the performance of the sampling techniques:
- **Naive Bayes** (`GaussianNB`)
- **XGBoost** (`XGBClassifier`)
- **LightGBM** (`LGBMClassifier`)
- **Perceptron** (`Perceptron`)
- **Gradient Boosting Classifier** (`GradientBoostingClassifier`)

## Workflow

1. **Data Balancing**: The dataset is balanced using SMOTE.
2. **Sampling**: Five different sampling techniques are applied to the balanced dataset.
3. **Model Training**: Each sampling technique is used to train all five models.
4. **Evaluation**: The models are evaluated on a test set using accuracy as the metric.
5. **Results Compilation**: An accuracy matrix is generated to compare the models and sampling techniques.

## Results

- The accuracy of each model is calculated for all sampling techniques.
- The best-performing sampling technique for each model is identified.

### Accuracy Matrix
The accuracy matrix shows the performance of each model under the different sampling techniques. 

### Best Sampling Techniques
The sampling technique that achieved the highest accuracy for each model is also reported.

## Files

- `Creditcard_data.csv`: The dataset used for this project.
- `accuracy_matrix.csv`: A CSV file containing the accuracy matrix.
- `best_combinations.csv`: A CSV file listing the best sampling technique for each model.
- `code.py`: The Python script used for the project.
