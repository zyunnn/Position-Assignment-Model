# Target Position Assignment Model

This repository is Zhi Yun Yap's solution to the programming project for Mid-To-Long-Term Quantitative Researcher. 

## 1. How to use
### 1.1 Setup
```
conda create -n project python=3.7
conda activate project
pip install -r requirements.txt
```
Only standard libraries are used in this project. 

### 1.2 Inference
To generate position assignment, run,
```
cd src
python3 run_assignment.py --file <FILENAME> --output_dir <DIR_PATH>
```
This generates position assignment and saves the results to `FILENAME`  under `output_dir`. 

To replicate results, run
```
python3 run_assignment.py --file target_position.csv --output_dir results
```


### 1.3 Evaluation
To evaluate position assignment, run
```
cd src
python3 run_evaluation.py --file <FILEPATH> --output_dir <DIR_PATH>
```
This prints a summary of the portfolio performance and saves corresponding charts under `output_dir`.

To replicate results, run
```
python3 run_evaluation.py --file target_position.csv --output_dir results
```


## 2. Methodology
### 2.1 Data Processing
* Load and reformat data from zip file
* Construct reversal signals at every minute
    - Bollinger band
    - Stochastic oscillator
    - MACD
    - Momentum 
* Fill missing price with last available price

### 2.2 Weighting Scheme
Taking momentum signal as an example, at the end of each minute, we rank stocks from lowest to highest trailing return. We then split the top/bottom half names into long/short bucket. For each stock in the long/short bucket, we can assign either (1) equal weighting, or (2) signal weighting to obtain the weights. 

1. Equal weighted

   Each stock in the long/short bucket is assigned equal proportion of the notional value
    
2. Signal weighted

   Normalized signal scores within long/short bucket by sum of signal scores in each bucket


### 2.3 Model
We start by constructing univariate model - use one reversal signal to weight position and evaluate the efficacy of individual signal to assigninig target position. 

The final position assignment model is a multivariate model -- take the average of **Bollinger band** and **Momentum** values as the new signal, and assign position using a signal-weighting scheme.

The model is developed with the following properties:

- Zero net exposure (measured in notional value) minute-by-minute
    - By normalizing signal scores within long/short bucket
- Zero overnight risk exposure
    - By liquidating all positions by market close


### 2.4 Key Assumption
When constructing and evaluating models, we made following assumptions:
* Frictionless market - ignore transaction cost, bid-ask spread, commission fee
* Able to rebalance portfolio and adjust position by the end of each minute interval
* Able to enter a position at `open` price in next minute


### 2.5 Evaluation
We evaluate overall performance of the position assignment model by evaluating the corresponding portfolio using
- Sharpe ratio
- Annualized volatility
- Max drawdown
- Max daily turnover
- Average number of assignments

We also inspect how portfolio PnL (overall, long/short bucket), return, volatility, turnover, and wealth curve evolves over time.



## 3. Implementation
The repository is structured as follows:
* Main modules
    - `data`
        - `dataloader`: handler to load and transform data
    - `models`
        - `base`: abstract class for position allocator
        -  `position_allocator`: concrete class for position assignment model
        - `backtest`: evaluate portfolio performance
    - `results`: store position assignment and evaluation results
* Others
    - `utils`: helper functions
    - `signals`: compute technical indicators as reversal signals

## 4. Future Work
* Evaluate sensitivity of portfolio performance with trading friction
    - The frictionless market is an ideal condition and can be unrealistic
    - Investigate how performance deterioriate under varying transaction cost from (e.g., 0bps - 30bps)
* Train a ML model (e.g. XGBoost) to select and combine technical indicators to generate more optimal position assignment
