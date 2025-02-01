# Regression-of-Used-Car-Prices
This repo is for participating in [this](https://www.kaggle.com/competitions/playground-series-s4e9/overview) Kaggle competition.

*"**Your Goal**: The goal of this competition is to predict the price of used cars based on various attributes."*

### Evaluation
Submissions are scored on the root mean squared error.

For each id in the test set, you must predict the price of the car. The file should contain a header and have the following format:

```
id,price
188533,43878.016
188534,43878.016
188535,43878.016
etc.
```

## Setup
This repo uses pre-commit hooks configured in `./.pre-commit-config.yaml` and a Conda environment configured in `environment.yml`. Here are the steps to set these up properly from this repos home folder:
1. Create an new Conda environment `conda env create -f environment.yml`
2. Activate the environment `conda activate Regression-Of-Used-Car-Prices`
3. Install pre-commit hooks `pre-commit install`

Pre-commit hooks can be stubborn when pushing to git without using the terminal (vscode, github desktop, etc). So be sure to commit and push through a terminal.  

If changes are made to `environment.yml` then update by running `conda env update --file environment.yml --prune`

## File Manifest
All models are genetated in the `./model/<model-type>` folders. The goal is to try to solve this with many different strategies. Models can predict on the data by running the `./model/model_predict.py` script (after changing the path to the model pickle file). Predictions are automatically formatted for Kaggle and stored in `./submissions`.

## Kaggle 
Results not submitted.

## TODO 
- Implement decision tree in XGBoost to leverage GPU support
- Build Logistic regression model
- Build Support Vector Machine model
- Build Neural Net Model

## Results
- 2027/3066
