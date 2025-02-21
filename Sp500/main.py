import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import traceback
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import os


class MacroMarketAnalysis:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = StandardScaler()

    def fetch_market_data(self):
        """Fetch S&P 500 data and calculate monthly returns"""
        try:
            print("Fetching S&P 500 data...")
            sp500 = yf.download("^GSPC", start=self.start_date, end=self.end_date)
            print(f"Dados brutos do S&P 500: {sp500.shape}")

            if sp500.empty:
                print("Erro: Não foram encontrados dados do S&P 500")
                return None

            print("Dados originais do S&P 500:")
            print(sp500.head())

            # Resample para dados mensais (último dia do mês)
            sp500_monthly = sp500['Close'].resample('ME').last()
            print("\nDados mensais:")
            print(sp500_monthly.head())

            # Calcular retornos
            sp500_returns = sp500_monthly.pct_change()
            sp500_returns = sp500_returns.dropna()

            print("\nRetornos calculados:")
            print(sp500_returns.head())

            # Criar DataFrame
            sp500_df = pd.DataFrame(sp500_returns)
            sp500_df.columns = ['SP500_Return']

            print(f"\nDados finais do S&P 500:")
            print(f"Shape: {sp500_df.shape}")
            print(f"Período: {sp500_df.index.min()} até {sp500_df.index.max()}")
            print("\nPrimeiras 5 linhas do DataFrame final:")
            print(sp500_df.head())

            return sp500_df

        except Exception as e:
            print(f"Error fetching market data: {e}")
            print("Traceback:")
            print(traceback.format_exc())
            return None

    def fetch_macro_data(self):
        """Fetch and process macroeconomic indicators"""
        try:
            print("Fetching macro data...")
            indicators = {
                'CPIAUCSL': 'CPI',
                'UNRATE': 'Unemployment',
                'FEDFUNDS': 'FedRate'
            }

            macro_data = {}
            for fred_code, name in indicators.items():
                try:
                    data = web.DataReader(fred_code, "fred", self.start_date, self.end_date)
                    macro_data[name] = data
                    print(f"Fetched {name} data: {len(data)} rows")
                except Exception as e:
                    print(f"Warning: Could not fetch {name} data: {e}")
                    continue

            print("\nCombinando dados macro...")
            macro_df = pd.concat(macro_data.values(), axis=1)
            macro_df.columns = macro_data.keys()

            print("Shape após concatenação inicial:", macro_df.shape)

            # Calcular inflação
            if 'CPI' in macro_df.columns:
                print("\nCalculando taxa de inflação...")
                macro_df['CPI'] = macro_df['CPI'].ffill()  # Forward fill CPI antes do cálculo
                macro_df['Inflation'] = macro_df['CPI'].pct_change() * 100
                print("Valores nulos na Inflation:", macro_df['Inflation'].isnull().sum())

            # Resample para fim do mês
            print("\nResampling para dados mensais...")
            macro_df = macro_df.resample('ME').last()
            print("Shape após resample:", macro_df.shape)

            # Forward fill após resample
            macro_df = macro_df.ffill()

            print("\nShape final dos dados macro:", macro_df.shape)
            print("Colunas disponíveis:", macro_df.columns.tolist())
            print("\nPrimeiras 5 linhas:")
            print(macro_df.head())

            return macro_df

        except Exception as e:
            print(f"Error in macro data processing: {e}")
            print("Traceback:")
            print(traceback.format_exc())
            return None

    def prepare_analysis_data(self, save_path="processed_data.csv"):
        """Prepare and combine all data for analysis and save to a CSV file."""

        market_returns = self.fetch_market_data()
        if market_returns is None:
            raise ValueError("Failed to fetch market data")

        macro_data = self.fetch_macro_data()
        if macro_data is None:
            raise ValueError("Failed to fetch macro data")

        # Normalizar os índices
        market_returns.index = pd.to_datetime(market_returns.index).normalize()
        macro_data.index = pd.to_datetime(macro_data.index).normalize()

        # Encontrar datas em comum
        common_dates = market_returns.index.intersection(macro_data.index)

        if len(common_dates) == 0:
            raise ValueError("No overlapping dates found between market and macro data")

        # Filtrar ambos os datasets para usar apenas as datas em comum
        market_data_aligned = market_returns.loc[common_dates]
        macro_data_aligned = macro_data.loc[common_dates]

        # Combinar os dados
        self.data = pd.concat([market_data_aligned, macro_data_aligned], axis=1)

        # Criar features adicionais
        self.data['Returns_Lag1'] = self.data['SP500_Return'].shift(1)
        self.data['Returns_Lag2'] = self.data['SP500_Return'].shift(2)
        self.data['Returns_MA3'] = self.data['SP500_Return'].rolling(window=3).mean()
        self.data['Returns_Vol3'] = self.data['SP500_Return'].rolling(window=3).std()

        # Forward fill e limpeza final
        self.data = self.data.ffill().dropna()

        # Salvar os dados processados em um arquivo CSV
        self.data.to_csv(save_path, index=True)
        print(f"Dados tratados salvos em: {os.path.abspath(save_path)}")

        return self.data

    def run_machine_learning_analysis(self):
        """Run multiple ML models and compare their performance"""
        if self.data is None:
            self.prepare_analysis_data()

        # Preparar features
        feature_cols = [col for col in self.data.columns if col != 'SP500_Return']
        X = self.data[feature_cols]
        y = self.data['SP500_Return']

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y)
        ridge_pred = ridge.predict(X_scaled)

        # Lasso Regression
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_scaled, y)
        lasso_pred = lasso.predict(X_scaled)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        rf_pred = rf.predict(X_scaled)

        # Feature Selection using RFE
        selector = RFE(ridge, n_features_to_select=5)
        selector.fit(X_scaled, y)

        # Collect results
        results = {
            'models': {
                'ridge': {'model': ridge, 'predictions': ridge_pred},
                'lasso': {'model': lasso, 'predictions': lasso_pred},
                'random_forest': {'model': rf, 'predictions': rf_pred}
            },
            'feature_importance': pd.DataFrame({
                'feature': feature_cols,
                'rf_importance': rf.feature_importances_,
                'selected': selector.support_
            })
        }

        return results

    def run_time_series_models(self):
        """Run time series specific models (ARIMA and LSTM)"""
        if self.data is None:
            self.prepare_analysis_data()

        # ARIMA Model
        arima = ARIMA(self.data['SP500_Return'], order=(2, 0, 2))
        arima_results = arima.fit()

        # LSTM Model
        X = self.data.drop('SP500_Return', axis=1).values
        y = self.data['SP500_Return'].values

        # Scale the data
        X_scaled = self.scaler.fit_transform(X)

        # Reshape for LSTM [samples, timesteps, features]
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Create and train LSTM model using Functional API
        inputs = Input(shape=(1, X_scaled.shape[1]))
        x = LSTM(50, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(30)(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)

        lstm = Model(inputs=inputs, outputs=outputs)

        # Use SparseCategoricalCrossentropy for classification tasks
        # Or keep 'mse' if this is a regression problem
        lstm.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )

        # Use validation_split instead of validation_data
        history = lstm.fit(
            X_lstm, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        return {
            'arima': arima_results,
            'lstm': {
                'model': lstm,
                'history': history.history
            }
        }

    def visualize_results(self, ml_results, ts_results):
        """Create comprehensive visualizations of all analyses"""
        # Usar estilo padrão do matplotlib
        plt.style.use('default')

        fig = plt.figure(figsize=(15, 12))

        # Plot 1: Feature Importance
        ax1 = plt.subplot(321)
        importance_df = ml_results['feature_importance']
        importance_df = importance_df.sort_values('rf_importance', ascending=True)
        ax1.barh(importance_df['feature'], importance_df['rf_importance'])
        ax1.set_title('Importância das Features (Random Forest)')
        ax1.set_xlabel('Importância')

        # Plot 2: Model Predictions Comparison
        ax2 = plt.subplot(322)
        for name, model_dict in ml_results['models'].items():
            ax2.scatter(self.data['SP500_Return'],
                        model_dict['predictions'],
                        alpha=0.5,
                        label=name)
        ax2.plot([self.data['SP500_Return'].min(), self.data['SP500_Return'].max()],
                 [self.data['SP500_Return'].min(), self.data['SP500_Return'].max()],
                 'k--', alpha=0.5)
        ax2.set_title('Retornos Previstos vs Reais')
        ax2.set_xlabel('Retornos Reais')
        ax2.set_ylabel('Retornos Previstos')
        ax2.legend()

        # Plot 3: Time Series of Returns and Major Indicators
        ax3 = plt.subplot(323)
        ax3.plot(self.data.index, self.data['SP500_Return'], label='Retornos')
        ax3.plot(self.data.index, self.data['Inflation'], label='Inflação')
        ax3.set_title('Retornos vs Indicadores Macro')
        ax3.set_xlabel('Data')
        ax3.set_ylabel('Valor (%)')
        ax3.legend()

        # Plot 4: LSTM Training History
        ax4 = plt.subplot(324)
        history = ts_results['lstm']['history']
        ax4.plot(history['loss'], label='Treino')
        ax4.plot(history['val_loss'], label='Validação')
        ax4.set_title('Histórico de Treinamento LSTM')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Erro (MSE)')
        ax4.legend()

        # Plot 5: Return Distribution
        ax5 = plt.subplot(325)
        ax5.hist(self.data['SP500_Return'], bins=30, edgecolor='black')
        ax5.set_title('Distribuição dos Retornos')
        ax5.set_xlabel('Retorno (%)')
        ax5.set_ylabel('Frequência')

        # Plot 6: Correlation Matrix
        ax6 = plt.subplot(326)
        corr_matrix = self.data[['SP500_Return', 'Inflation', 'Unemployment', 'FedRate']].corr()
        im = ax6.imshow(corr_matrix, cmap='RdBu')
        plt.colorbar(im, ax=ax6)
        ax6.set_xticks(range(len(corr_matrix.columns)))
        ax6.set_yticks(range(len(corr_matrix.columns)))
        ax6.set_xticklabels(corr_matrix.columns, rotation=45)
        ax6.set_yticklabels(corr_matrix.columns)
        ax6.set_title('Matriz de Correlação')

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    print("Iniciando análise...")
    analysis = MacroMarketAnalysis("1990-01-01", "2025-01-01")

    # Preparar dados
    data = analysis.prepare_analysis_data()

    # Executar análises
    print("\nExecutando modelos de machine learning...")
    ml_results = analysis.run_machine_learning_analysis()

    print("\nExecutando modelos de séries temporais...")
    ts_results = analysis.run_time_series_models()

    # Visualizar resultados
    fig = analysis.visualize_results(ml_results, ts_results)
    plt.show()

    # Imprimir resultados principais
    print("\nImportância das Features (Random Forest):")
    print(ml_results['feature_importance'].sort_values('rf_importance', ascending=False))

    print("\nPerformance do ARIMA:")
    print(ts_results['arima'].summary())
