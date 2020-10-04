from xgb_deploy.fmap import generate_fmap_from_pandas
from xgb_deploy.model import ProdEstimator
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from datetime import datetime
import xgboost as xgb
import pandas as pd
import numpy as np
import json


cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

# Replace space from column names to avoid JSON model dump errors
df.columns = [c.replace(' ', '_') for c in df.columns]

# Convert continuous fields to integer fields
for col in ['worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area']:
    df[col] = df[col].astype(int)

# Convert continuous fields to binary fields
for col in ['worst_smoothness', 'worst_compactness', 'worst_concavity']:
    df[col] = (df[col] < np.median(df[col])).astype(int)

# Generate fmap file
generate_fmap_from_pandas(df, 'fmap_pandas.txt')

feature_cols = [c for c in df.columns if c != 'target']

x_train, x_test, y_train, y_test = train_test_split(df[feature_cols], df['target'], test_size=0.33)

dtrain = xgb.DMatrix(data=x_train, label=y_train)
dtest = xgb.DMatrix(data=x_test, label=y_test)

regression_params = {
    'base_score': np.mean(y_train),
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'silent': 1
}

classification_params = {
    'base_score': 0.5,  # np.mean(y_train),
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': 1
}

num_boost_round = 10


def benchmark(params, pred_type):

        model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)

        model.dump_model(fout='xgb_model.json', fmap='fmap_pandas.txt', dump_format='json')

        with open('xgb_model.json', 'r') as f:
            model_data = json.load(f)

        estimator = ProdEstimator(model_data, pred_type, params['base_score'])

        predictions = model.predict(dtest)
        prediction_labels = [int(p >= 0.5) for p in predictions]
        prod_predictions = estimator.predict(x_test.to_dict(orient='records'))
        prod_labels = [int(p >= 0.5) for p in prod_predictions]

        tolerance = 0.001
        differences = [p - pp for p, pp in zip(predictions, prod_predictions)]
        matches = [np.abs(d) < tolerance for d in differences]

        print()
        print('Actual vs Prod Estimator Comparison')
        print('{} out of {} predictions match'.format(np.sum(matches), len(matches)))
        print('Mean difference between predictions: {}'.format(np.mean(differences)))
        print('Std dev of difference between predictions: {}'.format(np.std(differences)))
        print()
        print('Actual Estimator Evaluation Metrics')
        print('AUROC Score {}'.format(roc_auc_score(y_test, predictions)))
        print('Accuracy Score {}'.format(accuracy_score(y_test, prediction_labels)))
        print('F1 Score {}'.format(f1_score(y_test, prediction_labels)))
        print()
        print('Prod Estimator Evaluation Metrics:')
        print('AUROC Score {}'.format(roc_auc_score(y_test, prod_predictions)))
        print('Accuracy Score {}'.format(accuracy_score(y_test, prod_labels)))
        print('F1 Score {}'.format(f1_score(y_test, prod_labels)))
        print()
        print()
        test_data = x_test.head(1).to_dict(orient='records')
        print('Time Benchmarks for {} records with {} features using {} trees'.format(len(test_data), x_test.shape[1], num_boost_round))
        num_runs = 500
        runtimes = []

        for i in range(num_runs):
            start_time = datetime.now()
            estimator.predict(test_data)
            time_diff = (datetime.now() - start_time).total_seconds()
            runtimes.append(time_diff)

        avg = np.mean(runtimes)
        std = np.std(runtimes)
        print('Average {:.3e} seconds with standard deviation {:.3e} per {} predictions'.format(avg, std, len(test_data)))


if __name__ == '__main__':

    print('Benchmark regression modeling')
    print('=============================')
    benchmark(regression_params, 'regression')
    print()
    print()
    print('Benchmark classification modeling')
    print('=================================')
    benchmark(classification_params, 'classification')