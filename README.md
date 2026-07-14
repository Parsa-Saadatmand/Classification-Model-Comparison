# Titanic Survival Prediction — Model Comparison

Six classifiers, one preprocessing pipeline, same train/test split — trained and compared on the Titanic dataset to see which algorithm actually predicts survival best.

## Approach

Rather than picking one model and tuning it forever, I set up a single `ColumnTransformer` pipeline (median imputation + scaling for numeric features, most-frequent imputation + one-hot encoding for categoricals) and reused it across every model. That way the comparison is fair — every classifier sees exactly the same, leak-free data, and the only thing that changes is the algorithm itself.

Each model is wrapped in `GridSearchCV` (5-fold CV, optimizing F1) so the comparison reflects each algorithm's best version, not just its defaults.

## Models tested

| Model | What it's tuning |
|---|---|
| Logistic Regression | `C` (regularization strength) |
| Linear SVM | `C` |
| RBF-Kernel SVM | `C`, `gamma` |
| K-Nearest Neighbours | `n_neighbors`, weighting scheme |
| Decision Tree | `max_depth`, `min_samples_leaf`, `ccp_alpha` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf` |

Features used: `Age`, `Fare`, `SibSp`, `Parch`, `Pclass`, `Sex`, `Embarked`. `stratify=y` on the train/test split keeps the survival ratio consistent in both sets, since the classes aren't perfectly balanced to begin with.

## Results

F1-score is the main metric here rather than accuracy, since the target isn't perfectly balanced (roughly 62/38 split). Each model gets a full classification report and a confusion matrix. Random Forest comes out on top on the held-out test set, though the linear models aren't far behind — which says something about how much of the Titanic outcome really comes down to a handful of features (mostly sex and class).

## On the Decision Tree

Since it's the model most prone to overfitting here, I regularized it three ways: capping `max_depth`, raising `min_samples_leaf`/`min_samples_split` so splits need enough supporting data, and cost-complexity pruning (`ccp_alpha`) to trim back branches that add complexity without adding value.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook titanic_survival_prediction.ipynb
```

Needs a `titanic.csv` in the same folder — grab it from [Kaggle's Titanic competition](https://www.kaggle.com/c/titanic/data).

## Stack

pandas, numpy, scikit-learn, matplotlib, seaborn
