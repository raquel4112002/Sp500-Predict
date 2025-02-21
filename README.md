Relatório de Análise de Modelos para Previsão de Retorno do S&P 500

1. Introdução

O objetivo deste estudo foi analisar e comparar o desempenho de diferentes modelos de machine learning e séries temporais para a previsão do retorno do índice S&P 500. Utilizou-se uma base de dados com diversas variáveis econômicas e financeiras, como a taxa de desemprego, a inflação, as taxas de juros do Fed, o CPI (Índice de Preços ao Consumidor), e as médias móveis dos retornos, entre outras. O estudo focou-se em determinar o modelo que melhor se adequa à previsão de retorno do índice, utilizando os seguintes modelos: Ridge, Lasso, Random Forest, ARIMA, e LSTM.

2. Modelos Utilizados

Para alcançar o objetivo proposto, foram escolhidos e treinados os seguintes modelos:

Ridge Regression: É uma forma de regressão linear regularizada, onde se adiciona uma penalização ao modelo para reduzir a complexidade e evitar o overfitting. A principal vantagem do Ridge é a sua capacidade de lidar com multicolinearidade entre as variáveis independentes, ou seja, quando as variáveis explicativas estão fortemente correlacionadas. Neste caso, o modelo Ridge foi escolhido para capturar relações lineares entre as variáveis independentes e os retornos do S&P 500.

Lasso Regression: Similar ao Ridge, o Lasso também é uma técnica de regressão linear regularizada. A principal diferença é que o Lasso utiliza uma penalização L1, que força os coeficientes de algumas variáveis a se aproximarem de zero, efetivamente realizando seleção de características. O Lasso foi escolhido para avaliar o impacto da seleção de variáveis, podendo ser útil para identificar as variáveis mais relevantes na previsão do retorno.

Random Forest: Trata-se de um modelo de ensemble que utiliza árvores de decisão como base. Ele combina várias árvores de decisão para melhorar a precisão do modelo. A principal vantagem do Random Forest é que ele é capaz de modelar relações não lineares e complexas entre as variáveis e a variável alvo. Este modelo foi escolhido por sua capacidade de lidar com grandes quantidades de dados e sua robustez contra overfitting.

ARIMA (AutoRegressive Integrated Moving Average): ARIMA é um modelo clássico de séries temporais que pode ser utilizado para prever valores futuros com base em dados passados, levando em consideração dependências temporais. O ARIMA foi escolhido para capturar as dinâmicas temporais dos retornos do S&P 500.

LSTM (Long Short-Term Memory): LSTM é uma variação de redes neurais recorrentes (RNNs) projetada para lidar com longas dependências temporais, o que é crucial em séries temporais. Este modelo foi selecionado para testar a capacidade de redes neurais em capturar padrões temporais e não lineares nos dados.

3. Resultados Obtidos

Os resultados dos modelos foram avaliados em termos de três métricas principais: RMSE (Root Mean Squared Error), R² (Coeficiente de Determinação) e MAPE (Mean Absolute Percentage Error). A tabela a seguir apresenta os resultados obtidos para o conjunto de teste:
![image](https://github.com/user-attachments/assets/ff786ebc-972d-40d2-9b42-e06ef1fc0955)

![image](https://github.com/user-attachments/assets/65669b9e-6b95-4932-9867-35fec01403c3)

Análise dos Resultados:

Ridge:

O modelo Ridge apresentou os melhores resultados entre todos os modelos, com um RMSE de 0.0011, um R² de 0.9995 e um MAPE de 0.1951. O RMSE muito baixo indica que o modelo tem um desempenho muito bom na previsão dos retornos, com erro quadrático médio mínimo. O R² próximo de 1 sugere que o modelo explica praticamente toda a variabilidade dos dados. O MAPE, embora mais alto, ainda se mantém relativamente baixo, indicando que os erros percentuais são moderados.
Lasso:

O Lasso teve um desempenho significativamente pior em comparação ao Ridge, com um RMSE de 0.0495, R² negativo (-0.0080) e um MAPE de 2.3530. O modelo Lasso não foi capaz de ajustar os dados de forma eficaz, provavelmente devido à sua tendência de realizar uma forte seleção de variáveis, eliminando muitas delas e não capturando adequadamente as relações entre as variáveis e os retornos.
Random Forest:

O Random Forest obteve um RMSE de 0.0273, R² de 0.6935, e MAPE de 9.7898. Embora o RMSE e o R² mostrem que o modelo teve um desempenho razoável, o MAPE elevado indica que houve uma maior variabilidade nos erros percentuais. Este modelo é útil para lidar com dados não lineares, mas, no contexto desta tarefa, não foi tão preciso quanto o Ridge.
ARIMA:

O ARIMA obteve resultados semelhantes ao Lasso e Random Forest, com um RMSE de 0.0494, R² de -0.0070 e MAPE de 2.3827. O modelo ARIMA não conseguiu capturar adequadamente os padrões temporais dos dados, possivelmente devido à complexidade dos retornos do S&P 500 e à necessidade de ajustes mais refinados no modelo.
LSTM:

O modelo LSTM apresentou o pior desempenho entre todos os modelos, com RMSE de 0.1159, R² negativo (-4.3690) e MAPE extremamente alto (31.5150). Embora o LSTM seja ideal para capturar dependências temporais, os resultados sugerem que ele teve dificuldades para aprender os padrões dos dados financeiros, provavelmente devido a uma configuração inadequada de parâmetros ou à natureza dos dados.
4. Análise de Importância das Variáveis

Através do modelo Random Forest, foi possível calcular a importância das variáveis. A tabela a seguir mostra a importância das variáveis mais significativas na previsão do retorno do S&P 500:
![image](https://github.com/user-attachments/assets/5e07159c-5537-4462-a5b3-e4c3811316ef)

![image](https://github.com/user-attachments/assets/019241cb-d808-48be-9e95-d41b57cef732)

A variável Returns_MA3 (Média Móvel de 3 Meses dos Retornos) teve a maior importância, seguida por Returns_Vol3 (Volatilidade de 3 Meses dos Retornos). Essas variáveis estão relacionadas à dinâmica dos retornos Mensais do S&P 500 e, como esperado, têm um impacto significativo nas previsões do modelo.

5. Conclusão

Com base nos resultados, o modelo Ridge Regression mostrou-se o melhor modelo para a previsão do retorno do S&P 500, apresentando o menor erro (RMSE), o maior R² e um MAPE moderado. Este modelo é recomendado para este tipo de análise, devido à sua capacidade de capturar relações lineares e lidar com multicolinearidade entre as variáveis. O modelo Random Forest, embora robusto, não superou o Ridge, e os modelos Lasso, ARIMA e LSTM não conseguiram capturar os padrões de forma tão eficaz.


