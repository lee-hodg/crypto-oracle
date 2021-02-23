# Crypto Oracle

This app explores the use of Long-short term memory (LSTM) neural networks in predicting the price
of the cryptocurrency Bitcoin. 

## Project structure

### Notebooks

The `notebooks` dir contains a bunch of Jupyter notebooks, which were used when exploring the data
The initial notebook `1-Exploring-BTC-data.ipynb` loads the data and explores it, including
analysing the distribution of the returns and log returns and exploring the volatility of the returns.

The subsequent notebooks use a variety of models including univariate and multivariate (using both the close
prices and volume as features) LSTMs and Facebook Prophet.

I explored different scaling (standard, minmax, robust)
of the input data, predictions on the price series and predictions on the log returns.

The `utils.ipynb` notebook contains re-usable functions and helper code.

`Display_Results.ipynb` is a notebook that can load the Django training session models and display results
in the jupyter notebook for analyzing models trained by the web app.

### cryptooracle

This is the dir that holds the settings and other project files for the Django project.

### data

This contains the initial bulk CSV format stock data to be imported.

### model_weights

Here we store the trained weights of the Keras models so that we can re-load them and even update them on more data
later if we desire.

### scalers

Here we pickle the sci-kit learn scalers that were fit to the data so we can re-use them when inverting the data later.

### utils

Here we have a collection of re-usable functions.

### predictor

This Django app. This app has models that represent training sessions, stock OHLV and predictions.
The views and templates that form the web interface. The `management` directory contains scripts
for import stocks from CSV, pulling down the most recent stock data from the coin API
to update the database, training ML models and storing their weights and metadata and also
making forecasts over given data ranges with these models.

## The data

The initial bulk data was taken from [here](http://www.cryptodatadownload.com/data/gemini/)
This is minute-wise open, high, low, close, volume (OHLCV) data sourced
from the Gemini exchange since 2015. New data is pulled
periodically from [coin API](https://docs.coinapi.io/#latest-data).

Each stock is stored in a Django model `predictor.models.Stock`
that represents it in the database.

### Loading and updating the stock data

The initial load of the data can be done with

```python
./manage.py import_stocks
```

This will load all the CSV files from the `data` dir. 

To pull more recent data into the database use

```python
./manage.py update_stocks
```

This will source any missing future stocks from the Coin API.
(Note if many weeks or months are missing it might be better to refresh
the CSV files in the `data` dir and re-import, and only after that
run `update_stocks` for the very latest.)


## Training new models

You can train new models from the command line using
and specifying the training parameters. For example

```python
./manage.py train_model -S '2020-08-01' -E '2021-02-16' -X '2021-02-19' -P 5 -U 20 -I H```
```

The full list of arguments accepted can be oberseved in
`predictor.management.commands.train_model.Command.add_arguments`
but here a few are listed by frequently used or not:

Most commonly tweaked:

- `N`: the stock ticker name, e.g. `BTCUSD`
- `V`: is this an evaluation session or production session (see
  later in this readme for the difference)
- `S`: the start date from when to take stock date
- `E`: the end date
- `X`: are we updating the weights of an eXisting model on fresh data?
- `W`: the window length. This is the number of previous data points to consider
when making subsequent predictions in the series. E.g. with hourly resample
  data then a window length of 5 means use the previous 5 hours to predict the next hour's data point.
- `O`: output size. How many future data points do we want to try to predict?
- `I`: the interval, M, H, D are we going to train on minute by minute
stock data, hourly or daily?

 Less frequently needed to change:

- `F`: shuffle buffer size. A technical parameter to do with ensuring we shuffle randomly
the training data set before fitting the mode, so that we avoid bias.
- `T`: training split size, e.g. 80%.
- `U`: how many neurons in the LSTM hidden layers?
- `P`: epochs to train over
- `B`: batching data for efficiency
- `D`: the dropout probability. Dropout layers are used to regulaize the NN and this
parameter is the probabiliy of a node being dropped.
- `Z`:  the optimizer, e.g. adam
- `L`: loss function, e.g. mae
- `A`: hidden layer activation function, e.g. tanh
- 'C': scaler to use on the data, e.g. minmax, standard or robust


Training sessions are recorded in the model `predictor.models.TrainingSession`
This allows us to easily organize, re-load and re-train models.

This `train_model` command creates/loads an instance of that
model determined by the arguments passed. It builds the model in Keras,
loads and prepares the stock data from the database for training,
then fits the Keras model to the data. 

The training history is also saved to the Django model along with
other information such as the dates the training/test set was over.

The Keras model weights are saved to disk and the sci-kit learn
fitted scaler is pickled to disk too.

## Forecasting

After training a model, we can use it to generate price forcasts
by running

```python
./manage.py forecast -T <training_session_uuid>
```

This will store the forecasts into the `predictor.models.StockPrediction`
so they can be rapidly loaded by the web app.

## Evaluation vs production training sessions

Evaluation sessions are for evaluating the performance of a given model
(defined by a set of hyperparameters). Data will be from a given fixed date
range and split into training and test sets. We will graph the
performance against the training and test sets and compute metrics like
the MAE. 

The production training sessions however use ALL available data
to fit their weights and can be updated to the most bleeding edge
stock data we pull and then try to make forecasts into the future to
predict yet unseen prices.

### Trading strategy

We will also implement a basic trading strategy that involves going
long when the model forecasts a positive return and shorting when it forecasts
a negative return. We compare that against actual "correct moves"
(using the actual price dataset) and then do a hypothesis test to
check if our strategy is doing better than coin flipping.


## The web app

### Index page

The index page simply shows the BTCUSD price over recent dates along with
the traded volume underneath.

### Forecast page

This allows the selection of a date range and a production model
then shows the predictions the model makes vs the actual prices (for historic
prices) and any new prices for future dates

### Evaluation page

This allows the selection of evaluation models and shows their
performance on the test and train sets. 

Underneath we also plot the profits
over time for a simple trading strategy (described above)
along with the result of the significane test (do we reject the null 
hypothesis that we are not doing better than coin flipping? and
what is the p-value and t-score. Not that in the case
we don't reject the null and we are actually doing worse than coin flipping
the p-value could be low but the t-score is negative meaning
the sample mean was actually less than 0.5...lossing money :())


# Package management

[Poetry](https://python-poetry.org/docs/) is used to manage dependencies.

To set up the existing project for poetry first execute

```python
poetry init
```

This will generate the `pyproject.toml` file, with the dependencies and dev dependencies

For convenience during development we also have the following dev dependencies

```python
django-debug-toolbar
coloredlogs
Werkzeug
django-extensions
```

Next run

```
poetry install
```

Adding and removing packages with poetry:

```bash
poetry add X --dev # dev only
poetry add Y  # non -dev
poetry remove Z  # remove
```

Updating

```bash
poetry update X
```

Generating a regular `requirements.txt`

```bash
poetry export -f requirements.txt > requirements.txt --without-hashes
```

The location of the virtualenv set up by poetry can be found by running

```bash
poetry config --list 
```

# Heroku deploy and setup

See also [here](https://devcenter.heroku.com/articles/getting-started-with-python#deploy-the-app)

Create the heroku app with

The project uses [whitenoise](http://whitenoise.evans.io/en/stable/) for self-serving on static files
This involved adding the package with Poetry and then adding to the Django Middleware

```python
MIDDLEWARE = [
  # 'django.middleware.security.SecurityMiddleware',
  'whitenoise.middleware.WhiteNoiseMiddleware',
  # ...
]
```

We also use the [django-heroku](https://github.com/heroku/django-heroku) to care of a bunch
of things like `ALLOWED_HOSTS` and static files settings.


Create the app on Heroku

```bash
heroku create
```
The key can be generated with the command

```bash
python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'
```

Set some environment variables

```bash
heroku config:set DJANGO_SECRET_KEY="XXXX"
heroku config:set DJANGO_RUNTIME_ENVIRONMENT="production"
```

Now deploy with

```bash
git push heroku master
```


# AWS Deploy

First create the RDS instance

Install `awswebcli` on your system:

```bash
pip install --upgrade --user awsebcl
```

Add an AWS credentials profile to

```bash 
vim ~/.aws/credentials
```

For example

```bash 
[leehodg]
aws_access_key_id = XXXXX
aws_secret_access_key = XXXYYYZZZ
``

Then

```bash
eb init --profile <myprofile>
eb create crypto-oracle3 --single --instance-types t3.medium --profile <myprofile>
```

Will need to upgrade the instance type and set the env vars

```
RDS_DB_NAME
RDS_USERNAME
RDS_PASSWORD
RDS_HOSTNAME
RDS_PORT
DJANGO_SECRET_KEY
```

Note in AL2 the `PYTHONPATH` should already be set, and the method for installing
the postgres module has changed in `01_packages.config`
