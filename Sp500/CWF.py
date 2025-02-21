# -*- coding: utf-8 -*-
"""
MacroMarket Analysis Module
--------------------------
Este módulo implementa uma classe para análise de mercado macro utilizando
múltiplos modelos de machine learning e séries temporais.

Principais funcionalidades:
- Carregamento e preprocessamento de dados
- Implementação de modelos ML (Ridge, Lasso, Random Forest)
- Implementação de modelos de séries temporais (ARIMA, LSTM)
- Avaliação e comparação de modelos
- Visualização de resultados

Autores: [Seu Nome]
Data: [Data]
Versão: 1.0
"""

# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
import joblib
import pickle
import tensorflow as tf
# Importação dos modelos de machine learning
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
# Importação dos modelos de séries temporais
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import GridSearchCV



class MacroMarketAnalysis:
    """
        Classe principal para análise de mercado macro.

        Esta classe implementa uma suite completa de análise de dados financeiros,
        incluindo preprocessamento, modelagem e avaliação.

        Attributes:
            file_path (str): Caminho para o arquivo CSV com os dados
            data (pd.DataFrame): DataFrame contendo os dados carregados
            scaler (StandardScaler): Scaler para normalização dos dados
            X_train (pd.DataFrame): Dados de treino - features
            X_val (pd.DataFrame): Dados de validação - features
            X_test (pd.DataFrame): Dados de teste - features
            y_train (pd.Series): Dados de treino - target
            y_val (pd.Series): Dados de validação - target
            y_test (pd.Series): Dados de teste - target
        """

    def __init__(self, file_path):
        """
                Inicializa a classe MacroMarketAnalysis.

                Args:
                    file_path (str): Caminho para o arquivo CSV com os dados
                """

        self.file_path = file_path
        self.data = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.load_data()

    def load_data(self):
        """
        Carrega os dados do arquivo CSV e divide em conjuntos de treino/validação/teste.

        O método realiza as seguintes operações:
        1. Carrega o CSV usando pandas
        2. Separa features e target
        3. Realiza split temporal dos dados (sem shuffle)
        4. Primeiro split: 80% treino+validação, 20% teste
        5. Segundo split: 75% treino, 25% validação (do conjunto treino+validação)

        Raises:
            Exception: Se houver erro no carregamento ou processamento dos dados
        """
        try:
            print("Loading data from file...")
            self.data = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
            print(f"Data loaded successfully: {self.data.shape}")

            # Split features and target
            feature_cols = [col for col in self.data.columns if col != 'SP500_Return']
            X = self.data[feature_cols]
            y = self.data['SP500_Return']

            # Primeiro split: 80% train+val, 20% test
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            # Segundo split: 75% train, 25% validation
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, shuffle=False
            )

            print("Data split completed:")
            print(f"Training set: {self.X_train.shape}")
            print(f"Validation set: {self.X_val.shape}")
            print(f"Test set: {self.X_test.shape}")

        except Exception as e:
            print(f"Error loading data: {e}")
            print(traceback.format_exc())

    def optimize_ridge_model(self):
                """
        Otimiza o modelo Ridge usando GridSearchCV.

        Este método implementa:
        1. Pipeline com StandardScaler e Ridge
        2. Grid de hiperparâmetros para otimização
        3. Validação cruzada temporal
        4. Busca paralela dos melhores parâmetros

        Returns:
            sklearn.Pipeline: Melhor modelo encontrado após otimização
        """

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', Ridge())
                ])

                param_grid = {
                    'model__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
                    'model__fit_intercept': [True, False]
                }

                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )

                grid_search.fit(self.X_train, self.y_train)
                print(f"Best parameters: {grid_search.best_params_}")
                return grid_search.best_estimator_

    def create_ml_pipelines(self):
        """
        Cria pipelines para diferentes modelos de machine learning.

        Implementa três modelos:
        1. Ridge Regression com alpha=1.0
        2. Lasso Regression com alpha=0.1
        3. Random Forest com 100 estimadores

        Returns:
            dict: Dicionário contendo os pipelines dos modelos
        """
        pipelines = {
            'ridge': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Ridge(alpha=1.0))
            ]),
            'lasso': Pipeline([
                ('scaler', StandardScaler()),
                ('model', Lasso(alpha=0.1))
            ]),
            'random_forest': Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
        }
        return pipelines

    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Calcula métricas de performance para um modelo.

        Args:
            y_true (array-like): Valores reais
            y_pred (array-like): Valores previstos
            model_name (str): Nome do modelo sendo avaliado

        Returns:
            dict: Dicionário contendo as métricas calculadas
        """

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred)
        }
        print(f"\nMetrics for {model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper()}: {value:.4f}")
        return metrics

    def run_machine_learning_analysis(self):
        """
        Executa a análise completa usando múltiplos modelos de ML.

        Este método:
        1. Cria e treina múltiplos modelos
        2. Gera previsões para todos os conjuntos de dados
        3. Calcula métricas de performance
        4. Extrai importância das features do Random Forest

        Raises:
            ValueError: Se os dados não estiverem carregados

        Returns:
            dict: Resultados completos da análise, incluindo modelos,
                  previsões, métricas e importância das features
        """
        if self.data is None:
            raise ValueError("Data not loaded")

        pipelines = self.create_ml_pipelines()
        results = {'models': {}, 'metrics': {}}

        # Train and evaluate each model
        for name, pipeline in pipelines.items():
            # Train
            pipeline.fit(self.X_train, self.y_train)

            # Predictions
            train_pred = pipeline.predict(self.X_train)
            val_pred = pipeline.predict(self.X_val)
            test_pred = pipeline.predict(self.X_test)

            # Store results
            results['models'][name] = {
                'pipeline': pipeline,
                'predictions': {
                    'train': train_pred,
                    'val': val_pred,
                    'test': test_pred
                }
            }

            # Calcula métricas
            results['metrics'][name] = {
                'train': self.evaluate_model(self.y_train, train_pred, f"{name} (Train)"),
                'val': self.evaluate_model(self.y_val, val_pred, f"{name} (Validation)"),
                'test': self.evaluate_model(self.y_test, test_pred, f"{name} (Test)")
            }

        # Extrai importância das features do Random Forest
        rf_pipeline = pipelines['random_forest']
        rf_model = rf_pipeline.named_steps['model']
        results['feature_importance'] = pd.DataFrame({
            'feature': self.X_train.columns,
            'rf_importance': rf_model.feature_importances_
        })

        return results

    def create_lstm_model(self, input_shape):
        """
        Cria um modelo LSTM com Keras.

        Args:
            input_shape (tuple): Shape dos dados de entrada (lookback, features)

        Returns:
            tensorflow.keras.Model: Modelo LSTM compilado
        """

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(30),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def prepare_lstm_data(self, X, y, lookback=5):
        """
        Prepara sequências de dados para o LSTM.

        Args:
            X (array-like): Features
            y (array-like): Target
            lookback (int): Número de timesteps anteriores

        Returns:
            tuple: (X_sequences, y_sequences)
        """

        X_seq, y_seq = [], []
        for i in range(len(X) - lookback):
            X_seq.append(X[i:(i + lookback)])
            y_seq.append(y[i + lookback])
        return np.array(X_seq), np.array(y_seq)

    def run_time_series_models(self):
        """
    Executa modelos de séries temporais (ARIMA e LSTM).

    Este método implementa:
    1. Modelo ARIMA com ordem (2,0,2)
    2. Modelo LSTM com lookback de 5 períodos
    3. Normalização adequada dos dados
    4. Validação e avaliação dos modelos

    Returns:
        dict: Resultados dos modelos de séries temporais, incluindo:
            - Modelos treinados
            - Histórico de treinamento (LSTM)
            - Previsões
            - Métricas de performance

    Raises:
        ValueError: Se os dados não estiverem carregados
    """

        if self.data is None:
            raise ValueError("Data not loaded")

        results = {}

        # ARIMA Model
        try:
            arima = ARIMA(self.y_train, order=(2, 0, 2))
            arima_results = arima.fit()

            # Make predictions
            arima_pred = arima_results.forecast(len(self.y_test))
            results['arima'] = {
                'model': arima_results,
                'predictions': arima_pred,
                'metrics': self.evaluate_model(self.y_test, arima_pred, "ARIMA (Test)")
            }
        except Exception as e:
            print(f"Error in ARIMA modeling: {e}")

        # LSTM Model
        try:
            # Scale data
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(self.X_train)
            X_val_scaled = scaler_X.transform(self.X_val)
            X_test_scaled = scaler_X.transform(self.X_test)

            y_train_scaled = scaler_y.fit_transform(self.y_train.values.reshape(-1, 1))
            y_val_scaled = scaler_y.transform(self.y_val.values.reshape(-1, 1))
            y_test_scaled = scaler_y.transform(self.y_test.values.reshape(-1, 1))

            # Prepare sequences
            lookback = 5
            X_train_seq, y_train_seq = self.prepare_lstm_data(X_train_scaled, y_train_scaled, lookback)
            X_val_seq, y_val_seq = self.prepare_lstm_data(X_val_scaled, y_val_scaled, lookback)
            X_test_seq, y_test_seq = self.prepare_lstm_data(X_test_scaled, y_test_scaled, lookback)

            # Create and train model
            lstm = self.create_lstm_model((lookback, X_train_scaled.shape[1]))
            history = lstm.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                validation_data=(X_val_seq, y_val_seq),
                verbose=1
            )

            # Make predictions and inverse transform
            test_pred_scaled = lstm.predict(X_test_seq)
            test_pred = scaler_y.inverse_transform(test_pred_scaled)
            y_test_actual = scaler_y.inverse_transform(y_test_seq)

            results['lstm'] = {
                'model': lstm,
                'history': history.history,
                'predictions': test_pred,
                'metrics': self.evaluate_model(y_test_actual, test_pred, "LSTM (Test)")
            }
        except Exception as e:
            print(f"Error in LSTM modeling: {e}")

        return results

    def visualize_results(self, ml_results, ts_results):
        """
    Cria visualizações compreensivas de todas as análises.

    Gera 8 subplots diferentes:
    1. Importância das features (Random Forest)
    2. Comparação das previsões dos modelos
    3. Comparação das métricas de performance
    4. Série temporal dos retornos
    5. Histórico de treinamento do LSTM
    6. Previsões do LSTM vs valores reais
    7. Distribuição dos retornos
    8. Matriz de correlação

    Args:
        ml_results (dict): Resultados dos modelos de machine learning
        ts_results (dict): Resultados dos modelos de séries temporais

    Returns:
        matplotlib.figure.Figure: Figura contendo todos os plots
    """

        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))

        # Plot 1: Feature Importance
        ax1 = plt.subplot(331)
        importance_df = ml_results['feature_importance']
        importance_df = importance_df.sort_values('rf_importance', ascending=True)
        ax1.barh(importance_df['feature'], importance_df['rf_importance'])
        ax1.set_title('Feature Importance (Random Forest)')
        ax1.set_xlabel('Importance')

        # Plot 2: Model Predictions Comparison
        ax2 = plt.subplot(332)
        for name, model_dict in ml_results['models'].items():
            ax2.scatter(self.y_test,
                        model_dict['predictions']['test'],
                        alpha=0.5,
                        label=f"{name}\nR² = {ml_results['metrics'][name]['test']['r2']:.3f}")
        ax2.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 'k--', alpha=0.5)
        ax2.set_title('Predicted vs Actual Returns (Test Set)')
        ax2.set_xlabel('Actual Returns')
        ax2.set_ylabel('Predicted Returns')
        ax2.legend()

        # Plot 3: Performance Metrics Comparison
        ax3 = plt.subplot(333)
        metrics_data = []
        for model_name, metrics in ml_results['metrics'].items():
            metrics_data.append({
                'Model': model_name,
                'RMSE': metrics['test']['rmse'],
                'R²': metrics['test']['r2'],
                'MAPE': metrics['test']['mape']
            })
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.plot(x='Model', kind='bar', ax=ax3)
        ax3.set_title('Model Performance Metrics')
        ax3.set_ylabel('Score')
        plt.xticks(rotation=45)

        # Plot 4: Time Series of Returns
        ax4 = plt.subplot(334)
        ax4.plot(self.data.index, self.data['SP500_Return'], label='Actual Returns')
        ax4.set_title('S&P 500 Returns Over Time')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Return (%)')
        ax4.legend()

        # Plot 5: LSTM Training History
        ax5 = plt.subplot(335)
        history = ts_results['lstm']['history']
        ax5.plot(history['loss'], label='Train Loss')
        ax5.plot(history['val_loss'], label='Validation Loss')
        ax5.set_title('LSTM Training History')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss (MSE)')
        ax5.legend()

        # Plot 6: LSTM Predictions
        ax6 = plt.subplot(336)
        ax6.plot(self.y_test.index[-len(ts_results['lstm']['predictions']):],
                 ts_results['lstm']['predictions'],
                 label='LSTM Predictions')
        ax6.plot(self.y_test.index[-len(ts_results['lstm']['predictions']):],
                 self.y_test[-len(ts_results['lstm']['predictions']):],
                 label='Actual')
        ax6.set_title('LSTM Predictions vs Actual (Test Set)')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Return (%)')
        ax6.legend()

        # Plot 7: Return Distribution
        ax7 = plt.subplot(337)
        ax7.hist(self.data['SP500_Return'], bins=30, edgecolor='black')
        ax7.set_title('Return Distribution')
        ax7.set_xlabel('Return (%)')
        ax7.set_ylabel('Frequency')

        # Plot 8: Correlation Matrix
        ax8 = plt.subplot(338)
        corr_matrix = self.data.corr()
        im = ax8.imshow(corr_matrix, cmap='RdBu')
        plt.colorbar(im, ax=ax8)
        ax8.set_xticks(range(len(corr_matrix.columns)))
        ax8.set_yticks(range(len(corr_matrix.columns)))
        ax8.set_xticklabels(corr_matrix.columns, rotation=45)
        ax8.set_yticklabels(corr_matrix.columns)
        ax8.set_title('Correlation Matrix')

        plt.tight_layout()
        return fig

    # Add these methods to your MacroMarketAnalysis class
    def retrain_until_converged(self, best_model_name, ml_results, ts_results,
                                max_iterations=100,
                                error_threshold=0.01,  # 1% error margin
                                patience=5):
        """
        Retreina o melhor modelo iterativamente até atingir erro mínimo ou máximo de iterações.
        """
        print(f"\nIniciando retreinamento iterativo do modelo {best_model_name}...")

        # Preparar dados completos
        X_full = pd.concat([self.X_train, self.X_val, self.X_test])
        y_full = pd.concat([self.y_train, self.y_val, self.y_test])
        X_full = X_full.sort_index()
        y_full = y_full.sort_index()

        best_error = float('inf')
        best_model = None
        error_history = []
        no_improvement_count = 0

        for iteration in range(max_iterations):
            print(f"\nIteração {iteration + 1}/{max_iterations}")

            if best_model_name.lower() in ['ridge', 'lasso', 'random_forest']:
                # ML models
                pipeline = ml_results['models'][best_model_name.lower()]['pipeline']

                # Adjust hyperparameters based on previous errors
                if iteration > 0:
                    if best_model_name.lower() in ['ridge', 'lasso']:
                        current_alpha = pipeline.named_steps['model'].alpha
                        # Adjust alpha based on error trend
                        if len(error_history) >= 2 and error_history[-1] > error_history[-2]:
                            new_alpha = current_alpha * 1.5
                        else:
                            new_alpha = current_alpha * 0.8
                        pipeline.named_steps['model'].set_params(alpha=new_alpha)
                    elif best_model_name.lower() == 'random_forest':
                        current_n_estimators = pipeline.named_steps['model'].n_estimators
                        pipeline.named_steps['model'].set_params(n_estimators=current_n_estimators + 50)

                # Fit and predict
                pipeline.fit(X_full, y_full)
                y_pred = pipeline.predict(X_full)
                current_model = pipeline

            elif best_model_name.lower() == 'lstm':
                # Scale data
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_full_scaled = scaler_X.fit_transform(X_full)
                y_full_scaled = scaler_y.fit_transform(y_full.values.reshape(-1, 1))

                # Prepare sequences
                lookback = 5
                X_full_seq, y_full_seq = self.prepare_lstm_data(X_full_scaled, y_full_scaled, lookback)

                # Adjust LSTM architecture based on iteration
                n_units = 50 + (iteration * 10)  # Increase units progressively
                model = Sequential([
                    LSTM(n_units, return_sequences=True, input_shape=(lookback, X_full_scaled.shape[1])),
                    Dropout(0.2),
                    LSTM(n_units // 2),
                    Dropout(0.2),
                    Dense(1)
                ])

                # Adjust learning rate based on iteration
                lr = 0.001 * (0.95 ** iteration)  # Decrease learning rate progressively
                model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

                # Train with early stopping
                model.fit(
                    X_full_seq, y_full_seq,
                    epochs=50,
                    batch_size=32,
                    verbose=1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
                )

                # Predict
                y_pred_scaled = model.predict(X_full_seq)
                y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
                current_model = model

            elif best_model_name.lower() == 'arima':
                # Adjust ARIMA order based on iteration
                p = min(5, 2 + iteration // 10)
                d = min(2, iteration // 20)
                q = min(5, 2 + iteration // 10)

                arima = ARIMA(y_full, order=(p, d, q))
                fitted = arima.fit()
                y_pred = fitted.fittedvalues
                current_model = fitted

            # Calculate errors
            errors = np.abs((y_full - y_pred) / y_full)  # Percentage errors
            max_error = np.max(errors)
            mean_error = np.mean(errors)

            print(f"Erro máximo: {max_error:.4f} ({max_error * 100:.2f}%)")
            print(f"Erro médio: {mean_error:.4f} ({mean_error * 100:.2f}%)")

            error_history.append(max_error)

            # Check if this is the best model so far
            if max_error < best_error:
                best_error = max_error
                best_model = current_model
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Check convergence criteria
            if max_error <= error_threshold:
                print(f"\nConvergência atingida na iteração {iteration + 1}!")
                print(f"Erro máximo final: {max_error:.4f} ({max_error * 100:.2f}%)")
                return best_model, error_history, True

            # Check early stopping
            if no_improvement_count >= patience:
                print(f"\nParando treinamento - sem melhoria por {patience} iterações")
                print(f"Melhor erro máximo: {best_error:.4f} ({best_error * 100:.2f}%)")
                return best_model, error_history, False

        print("\nMáximo de iterações atingido sem convergência")
        print(f"Melhor erro máximo: {best_error:.4f} ({best_error * 100:.2f}%)")
        return best_model, error_history, False

    def plot_error_history(self, error_history):
        """
        Plota o histórico de erros durante o retreinamento.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(error_history, marker='o')
        plt.title('Evolução do Erro Máximo Durante Retreinamento')
        plt.xlabel('Iteração')
        plt.ylabel('Erro Máximo')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Configuração inicial
    file_path = "processed_data.csv"
    analysis = MacroMarketAnalysis(file_path)

    print("\nRunning machine learning models...")
    ml_results = analysis.run_machine_learning_analysis()

    print("\nRunning time series models...")
    ts_results = analysis.run_time_series_models()

    # Visualize initial results
    fig = analysis.visualize_results(ml_results, ts_results)
    plt.show()

    # Print final model comparison
    comparison_data = []

    # ML Models
    for model_name, metrics in ml_results['metrics'].items():
        comparison_data.append({
            'Model': model_name,
            'RMSE': metrics['test']['rmse'],
            'R²': metrics['test']['r2'],
            'MAPE': metrics['test']['mape']
        })

    # Time Series Models
    for model_name in ['arima', 'lstm']:
        if model_name in ts_results and 'metrics' in ts_results[model_name]:
            comparison_data.append({
                'Model': model_name.upper(),
                'RMSE': ts_results[model_name]['metrics']['rmse'],
                'R²': ts_results[model_name]['metrics']['r2'],
                'MAPE': ts_results[model_name]['metrics']['mape']
            })

    comparison_df = pd.DataFrame(comparison_data)
    print("\nAll Models Performance Comparison:")
    print(comparison_df.to_string(index=False))

    # Identify best model
    best_model_metrics = comparison_df.loc[comparison_df['RMSE'].idxmin()]
    best_model_name = best_model_metrics['Model']
    print(f"\nBest performing model: {best_model_name}")

    # Retrain best model iteratively until convergence
    final_model, error_history, converged = analysis.retrain_until_converged(
        best_model_name,
        ml_results,
        ts_results,
        max_iterations=100,  # Adjust as needed
        error_threshold=0.01,  # 1% error margin
        patience=5
    )

    # Plot error evolution
    analysis.plot_error_history(error_history)

    # Save final model
    if isinstance(final_model, Pipeline):
        joblib.dump(final_model, 'final_model_converged.joblib')
        print("Modelo final salvo em 'final_model_converged.joblib'")
    elif isinstance(final_model, tf.keras.Model):
        final_model.save('final_model_converged_lstm')
        print("Modelo final salvo em 'final_model_converged_lstm'")
    else:
        with open('final_model_converged_arima.pkl', 'wb') as f:
            pickle.dump(final_model, f)
        print("Modelo final salvo em 'final_model_converged_arima.pkl'")

    # Generate and save final predictions
    if isinstance(final_model, Pipeline):
        final_predictions = final_model.predict(analysis.data[analysis.X_train.columns])
    elif isinstance(final_model, tf.keras.Model):
        # Prepare data for LSTM
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(analysis.data[analysis.X_train.columns])
        lookback = 5
        X_seq = np.array([X_scaled[i:(i + lookback)] for i in range(len(X_scaled) - lookback)])
        final_predictions_scaled = final_model.predict(X_seq)
        final_predictions = scaler_y.inverse_transform(final_predictions_scaled).flatten()
    else:  # ARIMA
        final_predictions = final_model.fittedvalues

    # Create final results DataFrame
    final_results = pd.DataFrame({
        'Date': analysis.data.index[-len(final_predictions):],
        'Actual': analysis.data['SP500_Return'].values[-len(final_predictions):],
        'Predicted': final_predictions,
        'Error': analysis.data['SP500_Return'].values[-len(final_predictions):] - final_predictions
    })

    print("\nFinal Predictions Sample (last 5 records):")
    print(final_results.tail())

    # Save final results
    final_results.to_csv('final_model_predictions.csv')
    print("\nAnálise completa com modelo convergido.")