import os
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import re
import matplotlib.pyplot as plt
import kaggle
import traceback

from datetime import datetime
from scipy import stats
from itertools import combinations
from xgboost import XGBClassifier
from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model
from feature_engine.creation import MathFeatures
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import MeanMedianImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from pycaret.classification import predict_model, finalize_model, get_config, create_model
from sklearn.metrics import accuracy_score
from kaggle.api.kaggle_api_extended import KaggleApi

import warnings
warnings.filterwarnings("ignore")

class KaggleMLFlow:
    def __init__(self):
        pass
    
    def load_data(self, download_path='data/'):
        
        api = KaggleApi()
        api.authenticate()

        competition_name = 'inteligencia-artificial-2024-02'

        api.competition_download_file(competition_name, 'train.csv', path=download_path)
        api.competition_download_file(competition_name, 'test.csv', path=download_path)

        df = pd.read_csv(os.path.join(download_path, "train.csv"))
        prueba = pd.read_csv(os.path.join(download_path, "test.csv"))
        
        prueba_id = prueba["id"]
        df_id = df["id"]

        prueba = prueba.drop(columns=["id"])
        df = df.drop(columns=["id"])

        return df, prueba, df_id, prueba_id
    
    def get_columns(self, df, prueba, ct):
        for k in ct:
            df[k] = df[k].astype("O")
            prueba[k] = prueba[k].astype("O")

        le = LabelEncoder()
        df["Target"] = le.fit_transform(df["Target"])
        
        base_modelo = df.copy()
        base_modelo["Target"] = df["Target"].copy()
        base_modelo["Target"] = base_modelo["Target"].map(int)
        
        formatos = pd.DataFrame(base_modelo.dtypes).reset_index()
        formatos.columns = ["Variable", "Formato"]

        cuantitativas_bm = [x for x in formatos.loc[formatos["Formato"] != "object", "Variable"] if x not in ["id", "Target"]]
        categoricas_bm = [x for x in formatos.loc[formatos["Formato"] == "object", "Variable"] if x not in ["id", "Target"]]
        
        return base_modelo, cuantitativas_bm, categoricas_bm, le
        
    def process_data(self, base_modelo, prueba, cuantitativas_bm, categoricas_bm ):
        
        X = base_modelo.drop(columns=['Target'])
        y = base_modelo['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

        rare_encoder = RareLabelEncoder(tol=0.05, n_categories=2, variables=categoricas_bm)
        X_train = rare_encoder.fit_transform(X_train)
        X_test = rare_encoder.transform(X_test)
        prueba = rare_encoder.transform(prueba)

        one_hot_encoder = OneHotEncoder(drop_last=True, variables=categoricas_bm)
        X_train = one_hot_encoder.fit_transform(X_train)
        X_test = one_hot_encoder.transform(X_test)
        prueba = one_hot_encoder.transform(prueba)

        math_transformer = MathFeatures(variables=cuantitativas_bm, func=['sum', 'prod'])
        X_train = math_transformer.fit_transform(X_train)
        X_test = math_transformer.transform(X_test)
        prueba = math_transformer.transform(prueba)

        poly_transformer = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_train_poly = poly_transformer.fit_transform(X_train[cuantitativas_bm])
        X_test_poly = poly_transformer.transform(X_test[cuantitativas_bm])
        prueba_poly = poly_transformer.transform(prueba[cuantitativas_bm])

        X_train = np.hstack([X_train, X_train_poly])
        X_test = np.hstack([X_test, X_test_poly])
        prueba = np.hstack([prueba, prueba_poly])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        prueba = scaler.transform(prueba)

        X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        X_train['Target'] = y_train.reset_index(drop=True)

        X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        X_test['Target'] = y_test.reset_index(drop=True)
        
        prueba_df = pd.DataFrame(prueba, columns=[f'feature_{i}' for i in range(prueba.shape[1])])

        base_modelo = pd.concat([X_train, X_test], axis=0)
        
        return base_modelo, X_train, X_test, prueba_df
    
    def save_model(self, model, model_name, path):

        models_path = os.path.join(path, 'models')
        os.makedirs(models_path, exist_ok=True)

        file_name = os.path.join(models_path, f'{model_name}_model.pkl')
        file_name = os.path.normpath(file_name)

        with open(file_name, 'wb') as model_file:
            pickle.dump(model, model_file)

        print(f'Model {model_name} saved as {file_name}')

    def train_model(self, path, base_modelo, X_train, X_test, target_column, train_size=0.7, model_rank=1):
        base_modelo.reset_index(drop=True, inplace=True)

        exp_clf101 = setup(
            data=base_modelo,
            target=target_column,
            session_id=123,
            train_size=train_size,
            numeric_features=[f'feature_{i}' for i in range(X_train.shape[1] - 1)],  # última columna es 'Target'
            fix_imbalance=True
        )
        
        # Obtener los tres mejores modelos
        top_models = compare_models(include=['lightgbm', 'xgboost', 'rf', 'lr'], n_select=3)
        
        # Seleccionar el modelo según el rank
        trained_model = top_models[model_rank - 1]
        model_name = type(trained_model).__name__.lower()
        self.save_model(trained_model, model_name, path)

        param_grid = {
            'lgbmclassifier': {
                'max_depth': [3, 5, 7],
                'min_child_samples': [50, 100, 200],
                'num_leaves': [20, 31, 50],
                'learning_rate': [0.01, 0.05, 0.1],
            },
            'xgbclassifier': {
                'max_depth': [3, 5, 7],
                'min_child_weight': [5, 10, 20],
                'subsample': [0.6, 0.7, 0.8],
                'colsample_bytree': [0.6, 0.7, 0.8],
            },
            'randomforestclassifier': {
                'max_depth': [10, 20, 30],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 4, 6],
                'bootstrap': [True],
            },
            'logisticregression': {
                'penalty': ['l2', 'elasticnet'],
                'C': [0.01, 0.1, 1],
                'solver': ['lbfgs', 'saga'],
                'max_iter': [200, 500]
            }
        }

        tuned_model = tune_model(
            estimator=trained_model, 
            custom_grid=param_grid[type(trained_model).__name__.lower()],
            search_library='scikit-optimize', 
            search_algorithm='bayesian', 
            fold=5
        )
        
        tuned_model_name = f'{model_name}_tuned'
        self.save_model(tuned_model, tuned_model_name, path)
        
        predictions_test = predict_model(tuned_model, data=X_test)
        predictions_train = predict_model(tuned_model, data=get_config('X_train'))

        y_train = get_config('y_train')
        y_test = get_config('y_test')

        train_accuracy = accuracy_score(y_train, predictions_train["prediction_label"])
        print(f'Accuracy on training set for {model_name}: {train_accuracy}')
        test_accuracy = accuracy_score(y_test, predictions_test["prediction_label"])
        print(f'Accuracy on test set for {model_name}: {test_accuracy}')

        final_dt = finalize_model(tuned_model)
        
        return final_dt, train_accuracy, test_accuracy
    
    def perform_predictions(self, final_dt, prueba_df, le, prueba_id, path):
        predictions = predict_model(final_dt, data=prueba_df)

        predictions["Target"] = le.inverse_transform(predictions['prediction_label'])
        prueba_df["id"] = prueba_id.reset_index(drop=True)
        result = pd.DataFrame({
            'id': prueba_df["id"],
            'Target': predictions['Target']
        })
        
        print(f'Result value counts {result.Target.value_counts()}')
        print(f'Shape `{result.shape}')
        
        model_name = type(final_dt).__name__.lower()
        
        predictions_path = os.path.join(path, 'predictions')
        os.makedirs(predictions_path, exist_ok=True)
        
        result_file_name = f'{predictions_path}_{model_name}.csv'
        result.to_csv(result_file_name, index=False, sep=",")

        print(f'Result saved as {result_file_name}')
        
    def run_entire_work_flow(self, path, apply_feature_engineering=True, model_rank=1):
        try:
            # Cargar los datos
            df, prueba, df_id, prueba_id  = self.load_data()
            
            # ct = ['Gender', 'Displaced', 'Educational special needs', 'Debtor', 'Scholarship holder', 
            #       'International', 'Marital status', 'Nacionality', "Mother's occupation", "Father's occupation"]
            
            ct = ['Daytime/evening attendance', 'Displaced', 'Educational special needs', 'Debtor',
                    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Marital status',
                    'Application mode', 'Application order', 'Course', 'Previous qualification', 'Nacionality',
                    "Mother's qualification", "Father's qualification", "Mother's occupation",
                    "Father's occupation", ]

            # Obtener las columnas necesarias
            base_modelo, cuantitativas, categoricas, le = self.get_columns(df, prueba, ct)
            
            if apply_feature_engineering:
                base_modelo, X_train, X_test, prueba_df = self.process_data(base_modelo, prueba, cuantitativas, categoricas)
            else:
                # Sin ingeniería de características, simplemente dividir el conjunto de datos
                X = base_modelo.drop(columns=['Target'])
                y = base_modelo['Target']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
                X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
                X_train['Target'] = y_train.reset_index(drop=True)

                X_test = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
                X_test['Target'] = y_test.reset_index(drop=True)
                
                prueba_df = prueba.copy()

                # Base modelo sin ingeniería de características
                base_modelo = pd.concat([X_train, X_test], axis=0)
            
            final_dt, accuracy_train, accuracy_test = self.train_model(path, base_modelo, X_train, X_test, 'Target', model_rank=model_rank)
            
            self.perform_predictions(final_dt, prueba_df, le, prueba_id, path)
            
            return {'success': True, 'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test}
        
        except Exception as e:
            error_trace = traceback.format_exc()
            print(error_trace)  # Imprimir la traza completa para más detalles
            return {'success': False, 'message': str(e), 'traceback': error_trace}

flow = KaggleMLFlow()
result = flow.run_entire_work_flow("C:/Users/jesco/OneDrive - Universidad Santo Tomás/Documentos/Python/T3", apply_feature_engineering=True, model_rank=1)
print(result)