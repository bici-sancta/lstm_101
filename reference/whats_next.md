### exploratory analysis to understand nature / behavior of long sort term memory (LSTM) recurrent neural network (RNN) application and behavior for time-series modeling

* financial market time-series data
* US equity market data
    * initiate with S&P500 historical prices
        * available daily records from Jna-1928
        * Date,Open,High,Low,Close,Adj Close,Volume
        
#### basic concepts



#### what is done

##### function
- basic LSTM time-series forecast
- 1 or 2 column inputs as explanatory variables
- 1-step lagging (only) ?
- 1-step forward prediction
- n-future time steps
- n-seq number of steps for input sequence
- n-layers modifiable number of LSTM + Dropout layers
- single valued output (dense layer output)
- multi-feature input seems to be working - needs to be verified


##### output / plotting
- basic plot of observed (target variable) and predicted values


#### what is yet to be done

* verify all arrays are correctly lagged
* add hyperparameter tuning
* more sophisticated or more options for future modeling of explanatory variables
* increase number of covariates

##### output / plotting
- plot covariates

#### sources of information / inspiration

