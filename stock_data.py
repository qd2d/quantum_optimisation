#####################################################################################################################
# This code below was written by Davie John Mtsuko. The purpose of the code is to create an app for portfolio       #
# optimization using a local NumpyEigenSolver and a cloud based IBM quantum computer.                               #
#####################################################################################################################

"""
After running this code, you will get a graphical user interface (GUI) where you enter the symbols for the companies
you are considering to study and choose some of them for investing. You will also get a plot of the covariance
matrix.Enter the symbol for example as AAPL/MSFT/GOOG/IBM separated by /. Next you must enter a trading period start
date for example as 2013/1/1. Click button save trade date and close the GUI. You will get a plot of trade data from
the start date to present day in the plots window. The choice of companies to choose to invest in will be displayed
in the Python Console.
"""
from pathlib import Path
from qiskit.result import QuasiDistribution
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
import api
import json
from fastapi import FastAPI, Request, Form
from flask import Flask
from flask import request

import PIL
from PIL import Image
mywidth=100
myheight=100

import os
from multiprocessing import Process
import asyncio
from fastapi import FastAPI
import uvicorn
from twilio.rest import Client

account = "your account"
token = "your token"
client = Client(account, token)



#Import SQLite3 database class

from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()
import sqlite3


#The following section imports classes in the kivy library that are useful in app development

import contextlib
import kivy.uix.screenmanager
#import numpy

####
from kivy.config import Config
Config.set('kivy', 'exit_on_escape', '0')

from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.core.window import Window
####

from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
import sdl2

#The following section imports classes from the qiskit library that are useful in solving optimization problems on local
#machine or on a real IBM quantum computer.

from qiskit import IBMQ
from qpo.vqe.vqe_solver import VQESolver
from qiskit.utils import algorithm_globals
from qiskit import BasicAer
from qiskit.utils import QuantumInstance

from qiskit import Aer
#from qiskit.aqua.operators import *
#from qiskit.aqua.operators.expectations import PauliExpectation,AerPauliExpectation,MatrixExpectation
from qiskit.algorithms.minimum_eigen_solvers import VQE, QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import TwoLocal

from qiskit.finance.applications.ising import portfolio
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.finance.data_providers import RandomDataProvider
from qiskit.finance.data_providers import YahooDataProvider, DataOnDemandProvider
from qiskit.opflow.primitive_ops import PrimitiveOp
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_optimization.converters import QuadraticProgramToQubo
#from qiskit.aqua.operators import (OperatorBase, ExpectationBase, ExpectationFactory, StateFn,
#                                   CircuitStateFn, LegacyBaseOperator, ListOp, I, CircuitSampler)

from qiskit.algorithms.optimizers import COBYLA

#Import numpy and matplotlib library for handling mathematics and plotting graphs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib.use('agg') #('TkAgg') Removed due to bug with Tinkter
import datetime

#Import the pandas and time definition libraries
import pandas as pd
from pandas_datareader import data as web
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

#Import tools for downloading stock exchange trade data
import alpaca_trade_api as tradeapi
import yfinance

#import table, statistical and plotting tools
import csv
from statistics import mean
import seaborn as sn
import plotly.graph_objs as go

#Option for classes and methods
from abc import ABC, abstractmethod

# Get the IBM quantum computing service provider and print the list of available simulators and devices.
# Use Aer to access simulators or access the real IBMQ quantum computer

provider= Aer
#provider = IBMQ.load_account()
for backend in provider.backends():
   print(backend)
name = "statevector_simulator"
#global backend
backend = provider.get_backend(name)
print(backend.name())

def quantum_instances(backend):
    #global backend
    backend.configuration
    QuantumInstance(backend)
    return backend

def configuration(backend):
    qi=quantum_instances(backend)
    config=qi.configuration
    print(config)
    return config

config=configuration(backend)

def break_period_in_dates_list(start_date, end_date, days_per_step):
    '''Break period between start_date and end_date in steps of days_per_step days.'''
    step_start_date = start_date
    delta = timedelta(days=days_per_step)
    dates_list = []
    while end_date > (step_start_date + delta):
        dates_list.append((step_start_date, step_start_date + delta))
        step_start_date += delta
    dates_list.append((step_start_date, end_date))
    return dates_list

def format_timestep_list(timestep_list):
    '''Format dates in ISO format plus timezone. Note that first day starts at 00:00 and last day ends at 23:00hs.'''
    for i, d in enumerate(timestep_list):
        timestep_list[i] = (d[0].isoformat() + '-04:00', (d[1].isoformat().split('T')[0] + 'T23:00:00-04:00'))
    return timestep_list

def get_df_from_barset(barset):
    '''Create a Pandas Dataframe from a barset.'''
    df_rows = []
    for symbol, bar in barset.items():
        rows = bar.__dict__.get('_raw')
        for i, row in enumerate(rows):
            row['symbol'] = symbol
        df_rows.extend(rows)
    return pd.DataFrame(df_rows)

def download_data(aps, symbols, start_date, end_date, filename='data.csv'):
    '''Download data from REST manager for list of symbols, from start_date at 00:00hs to end_date at 23:00hs,
    and save it to filename as a csv file.'''
    timesteps = format_timestep_list(break_period_in_dates_list(start_date, end_date, 10))
    df = pd.DataFrame()
    for timestep in tqdm(timesteps):
        barset = aps.get_barset(symbols, '5Min', limit=1000, start=timestep[0], end=timestep[1])
        df = df.append(get_df_from_barset(barset))
        time.sleep(0.1)
    df.to_csv(filename)

# Access database
connection = sqlite3.connect("app.db")
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

cursor.execute("""
    SELECT id FROM strategy WHERE  name = 'optimum_portfolio_moving_average'
    """)
strategy_id = cursor.fetchone()['id']
print(strategy_id)

cursor.execute("""
        SELECT name FROM strategy WHERE  id = '19'
""")
strategy = cursor.fetchone()['name']
print(strategy)

#Calling from sqlite database the symbols we want to put in our portfolio for optimization
cursor.execute("""
    SELECT symbol, company FROM stock
    join stock_strategy on stock_strategy.stock_id= stock.id
    where stock_strategy.strategy_id = ?
""",(strategy_id,))

stocks =cursor.fetchall()

#Calling from database.json the symbols we want to put in our portfolio for optimization
with open('database.json') as f:
    symbols_data = json.load(f)

symbols = [symbols_data['1'], symbols_data['2'],symbols_data['3'],symbols_data['4'],symbols_data['5'], symbols_data['6']]  # [stock['symbol'] for stock in stocks]
names = [stock['company'] for stock in stocks]
connection.commit()
""""
# Call method.
aps = tradeapi.REST(key_id='your ID',
                    secret_key='your key',
                    base_url='https://api.alpaca.markets')
                    #base_url='https://data.alpaca.markets')
download_data(aps=aps,
              symbols=['MSFT', 'GOOG','AMZN', 'TSLA'],
              start_date=datetime.strptime('1/10/21', '%d/%m/%y'),
              end_date=datetime.strptime('29/10/21', '%d/%m/%y'),
              filename='test.csv')
# opening the CSV file
with open('test.csv', mode='r') as file:
    # reading the CSV file
    csvFile = csv.reader(file)

    # displaying the contents of the CSV file
    matrix=[]
    for lines in csvFile:
        matrix.append(lines)
    print(matrix)
"""
# set number of assets (= number of qubits)
global num_assets
num_assets = 6
"""
#tickers=  ['FB','AAPL', 'TSLA', 'MSFT', 'GOOG','AMZN']

#Option for getting variance matrices
u=[]
for i in range(1, 30):
    ui=matrix[i][5]
    uf=float(ui.strip().strip("'"))
    u.append(uf)
print(u)

v=[]
for i in range(625, 654):
    vi=matrix[i][5]
    vf=float(vi.strip().strip("'"))
    v.append(vf)
print(v)
x=[]
for i in range(845, 874):
    xi=matrix[i][5]
    xf=float(xi.strip().strip("'"))
    x.append(xf)
print(x)
y=[]
for j in range(1157, 1186):
    yi = matrix[j][5]
    yf = float(yi.strip().strip("'"))
    y.append(yf)
print(y)
print(len(matrix))

# Covariance
def cov(x, y):
    xbar, ybar = mean(x), mean(y)
    for i in range (1, len(x)):
        for j in range (1, len(y)):
            return np.sum((float(x[i]) - xbar)*(float(y[j]) - ybar))/(len(x) - 1)

# Calculate covariance matrix
print(cov(x,y))
# Covariance matrix
X=np.array([u, v, x,y])
covMatrix=np.cov(X, bias=True)
print(covMatrix)
"""
def store_intermediate_result(eval_count, parameters, mean, std):
    history['eval_count'].append(eval_count)
    history['parameters'].append(parameters)
    history['mean'].append(mean)
    history['std'].append(std)

algorithm_globals.random_seed = 1234

def gen_cov(date):
    # Generate expected return and covariance matrix from (random) time-series
    # stocks = [("TICKER%s" % i) for i in range(num_assets)]
    # stocks2 = ['AAPL', 'GOOG','MSFT', 'TSLA']
    global stocks2
    #stocks2 = ['FB', 'AAPL','TSLA', 'MSFT', 'GOOG', 'AMZN']
    #stocks2 = list(map(str,bbbb.split('/')))
    stocks2 = symbols
    print('Reading symbols from SQL database that were selected on the trading web UI:', stocks2)
    # data1 = YahooDataProvider(tickers=stocks2, start=datetime(2021,10,1), end=datetime(2021,10,29))
    global seed
    seed = 123
    """
    For Cryto markets
    data1=[]
    for coin in names:

        coinname = coin[0].lower() + coin[1:]

        coindata = cg.get_coin_ohlc_by_id(id=coinname, vs_currency= 'usd', days=365)
        print(coinname)

        #dataarray = np.array(coindata)

        data1frame = pd.DataFrame(coindata)
        df = data1frame
        new_cols = ["datetime", "o", "h","l", "c"]

        df[new_cols] = df

        df.drop(df.columns[[0,1,2,3,4]], axis = 1, inplace = True)

        df['index'] = df.groupby('datetime', sort=False).ngroup() + 1
        print(df)

        #print(data1m)
        vector = df['o'].values

        print(vector)

        data1.append(vector)

    """
    with open('database.json') as f:
        symbols_data = json.load(f)
    stockStartDate = symbols_data['7'] #'2022-08-01'
    global enddate
    enddate = symbols_data['8']
    data1 = YahooDataProvider(tickers=stocks2, start=datetime(int(list(map(str, startdate.split('-')))[0]),int(list(map(str, startdate.split('-')))[1]),int(list(map(str, startdate.split('-')))[2])),end = datetime(int(list(map(str, enddate.split('-')))[0]),int(list(map(str, enddate.split('-')))[1]),int(list(map(str, enddate.split('-')))[2])) )#end = datetime(2022,11,10)

    data1.run()
    global mu
    mu = data1.get_period_return_mean_vector()
    print(mu)
    #sigma= covMatrix
    global sigma
    sigma = data1.get_period_return_covariance_matrix()
    sn.set(style="ticks", context="talk")

    plt.style.use("dark_background") #"default"
    plt.title('Covariance Matrix', fontsize=12)
    sn.heatmap(sigma, annot=False, fmt='g', xticklabels=stocks2, yticklabels=stocks2) #fmt='.2g' or annot=True, fmt='g', stocks2
    path_to_filem1 = 'covariance.png'
    pathm1 = Path(path_to_filem1)
    if pathm1.is_file():
        os.remove('covariance.png')
        time.sleep(5)
    else:
        pass
    plt.savefig('covariance.png')
    time.sleep(5)
    #plt.show()
    plt.close()

    #img=PIL.Image.open('covariance.png')
    #img=img.resize((mywidth,myheight),PIL.Image.ANTIALIAS)
    #img.save('mobile_covariance.png')
    #plt.show()

    # plot sigma
    sn.set_style("dark")
    plt.imshow(sigma, interpolation='nearest')
    #plt.show()
    plt.close()
    global q
    global budget
    global penalty
    q = 0.25  # set risk factor
    print('risk is', q)
    budget = num_assets // 2 # set budget div by
    penalty = num_assets  # set parameter to scale the budget penalty term

    qubitOp, offset = portfolio.get_operator(mu, sigma, q, budget, penalty)
    print(qubitOp)
    global portfolio2
    portfolio2 = PortfolioOptimization(mu, sigma, q, budget)
    qp = portfolio2.to_quadratic_program()
    print(qp)
    exact_mes = NumPyMinimumEigensolver()
    exact_eigensolver = MinimumEigenOptimizer(exact_mes)

    global result

    #result = exact_eigensolver.solve(qp)
    #print(result)
    #print_result(result)

    #####----VQE testing start-----#####
    backend = Aer.get_backend('statevector_simulator')
    seed = 50

    cobyla = COBYLA()
    cobyla.set_options(maxiter=500)
    #num_qubits = 6
    qubitOp, offset = portfolio.get_operator(mu, sigma, q, budget, penalty)

    ry = TwoLocal(qubitOp.num_qubits, 'ry', 'cz', reps=3, entanglement='full')


    quantum_instance = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
    vqe_meas = VQE(ry, cobyla, quantum_instance=quantum_instance)
    vqe_meas.random_seed = seed



    #result = vqe.run(quantum_instance)
    vqe = MinimumEigenOptimizer(vqe_meas)
    result = vqe.solve(qp)

    print_result(result)
    #####----VQE testing end-----#####

"""
    from qiskit.utils import algorithm_globals
    algorithm_globals.random_seed = 1234




    #slsqp=SLSQP()
    cobyla = COBYLA()
    optimizer=cobyla.set_options(maxiter=250)
    #global num_qubits
    #num_qubits =4
    #pauli_terms = ['IIIIIIIZ', 'IIIIIIZI', 'IIIIIZII']
    #pauli_weights = [504.0, 1008.0, 2016.0]
    #pauli_dict = {'paulis': [{"coeff": {"imag": 0., "real": pauli_weights[i]}, "label": pauli_terms[i]} \
    #                          for i in range(len(pauli_terms))]}
    #Hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)
    global reps
    reps = 3
    maxiter=10
    Nq=1
    vqe = VQESolver()
    vqe.qp(Cov = sigma)
    dddd=vqe.to_ising()
    print(dddd)

    # Prepare QuantumInstance

    # Select the VQE parameters
    N = sigma.shape[0]
    print(N)
    operator_1 = PrimitiveOp(np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]]))

    #ansatz = TwoLocal(N * Nq, ['ry', 'rz'], 'cz', 'full', reps )
    #print(ansatz)
    #slsqp = SLSQP(maxiter)
    #vqe.vqe_instance(ansatz, slsqp, quantum_instance, None, None)
    #vqe_res= vqe.solve()
    #print(vqe_res)

    #ry = TwoLocal(N * Nq, ['ry'], 'cz', 'full', reps)

    #global var_form
    #var_form = ry

    #quantum_instance = quantum_instances(backend)
    #vqe_mes = VQE(var_form, cobyla, None, None, None, True,1, None,quantum_instance, None)
    #vqe_mes = VQE(var_form=var_form, optimizer=cobyla, quantum_instance=quantum_instance)
    #print(quantum_instance)
    #print(backend.configuration())
    #vqe= MinimumEigenOptimizer(vqe_mes)
    #print(vqe)

    #result2=vqe.solve(qp)
    #print_result(result2)

    #TwoLocal()
    #print(ry)

    #quantum_instance = QuantumInstance(backend, shots= 128, initial_layout=None,optimization_level=3) #8192

    qubit_op = qubitOp.to_opflow().to_matrix_op()

    #n_qubits=num_assets
    n_qubits = 6 #Use 2
    initial = QuantumCircuit(n_qubits)
    # add any gate you want in the circuit, for example :
    initial.h(0)
    initial.cx(0, 1)
    optimizer = COBYLA()
    quantum_instance = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
    print('Quantum instance:',quantum_instance)
    #operator = operator_1
    #print('operator is:',operator)
"""
"""
    algorithm_globals.random_seed = 10598
    optimizer = COBYLA()
    qaoa_mes = QAOA(optimizer, reps=4, quantum_instance=quantum_instance)#3
    print(qaoa_mes)

    qaoa = MinimumEigenOptimizer(qaoa_mes)
    result2 = qaoa.solve(qp)
    print(result2)
    print_result(result2)
    return backend, qubitOp, offset
"""
def plotstock(source):
    #plt.style.use('fivethirtyeight')
    plt.style.use("dark_background") #plt.style.use("default") or "default"
    with open('database.json') as f:
        symbols_data = json.load(f)
    stockStartDate = symbols_data['7'] #'2022-08-01'
    global enddate
    enddate = symbols_data['8']
    today = enddate#'2022-11-10'#datetime.today().strftime('%Y-%m-%d')
    print("Optimization time window is:", stockStartDate,"-", today)

    df2=pd.DataFrame()

    for stock in stocks2:
        yfinance.pdr_override()
        #df2[stock] = web.DataReader(stock, data_source=source, start=stockStartDate, end = today)['Adj Close']# end=today
        df2[stock] = pdr.get_data_yahoo(stock, start=stockStartDate, end=today)['Adj Close']  # Replace with previous when yahoo is fixed
    print(df2)

    title = 'Portfolio Adj. Close Price History'
    global my_stocks
    my_stocks=df2

    for c in my_stocks.columns.values:
        plt.plot(my_stocks[c], label = c)

    plt.title('Historical Stock Data',fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Adj. Price USD ($)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.gcf().autofmt_xdate()
    plt.yticks(fontsize=12)
    plt.yscale('linear')
    plt.legend(my_stocks.columns.values, loc='upper left')
    path_to_file0 = 'stock.png'
    path0 = Path(path_to_file0)
    if path0.is_file():
        os.remove('stock.png')
        time.sleep(5)
    else:
        pass
    plt.savefig('stock.png')
    time.sleep(5)
    plt.close()
    #plt.show()

def index_to_selection(i, num_assets):
    s = "{0:b}".format(i).rjust(num_assets)
    x = np.array([1 if s[i] == '1' else 0 for i in reversed(range(num_assets))])
    return x

def print_result(result):
    global selection
    global value
    selection = result.x
    value = result.fval

    #selection = sample_most_likely(result.eigenstate)
    #value = portfolio.portfolio_value(selection, mu, sigma, q, budget, penalty)

    print('Optimal: selection {}, value {:.4f}'.format(selection, value))
    #eigenstate = result.min_eigen_solver_result.eigenstate
    eigenstate = result.min_eigen_solver_result.eigenstate

    #eigenvector = eigenstate if isinstance(eigenstate, np.ndarray) else eigenstate.to_matrix()
    eigenvector = eigenstate if isinstance(eigenstate, QuasiDistribution) else eigenstate#.to_matrix()

    probabilities = np.abs(eigenvector) ** 2
    i_sorted = reversed(np.argsort(probabilities))
    print('\n----------------- Full result ---------------------')
    print('selection\tvalue\t\tprobability\texpected value')
    print('---------------------------------------------------')
    for i in i_sorted:
        x = index_to_selection(i, num_assets)
        value = portfolio.portfolio_value(x, mu, sigma, q, budget, penalty)
        exp_value=portfolio.portfolio_expected_value(x,mu) #CAN BE REMOVED
        probability = probabilities[i]
        print('%10s\t%.4f\t\t%.4f\t%.4f' % (x, value, probability,exp_value))

    return selection


def gen_algotrade():
    yfinance.pdr_override()

    # download dataframe
    # data2 = pdr.get_data_yahoo("MSFT", start="2021-08-01", end="2021-08-05")
    """
    data1 = yfinance.multi.download(tickers="AAVMY", period='60d', interval='15m', rounding=True)
    data2 = yfinance.multi.download(tickers="PHG", period='60d', interval='15m', rounding=True)
    data3 = yfinance.multi.download(tickers="ING", period='60d', interval='15m', rounding=True)
    data4 = yfinance.multi.download(tickers="ESNT", period='60d', interval='15m', rounding=True)
    data5 = yfinance.multi.download(tickers="IBM", period='60d', interval='15m', rounding=True)
   """

    data1 = yfinance.multi.download(tickers=stocks2[0], period='30d', interval='15m', rounding=True) #"SEDG"
    data2 = yfinance.multi.download(tickers=stocks2[1], period='30d', interval='15m', rounding=True) #"SOL"
    data3 = yfinance.multi.download(tickers=stocks2[2], period='30d', interval='15m', rounding=True)#"SPWR"
    data4 = yfinance.multi.download(tickers=stocks2[3], period='30d', interval='15m', rounding=True)#"DQ"
    data5 = yfinance.multi.download(tickers=stocks2[4], period='30d', interval='15m', rounding=True)#"IBM"
    data6 = yfinance.multi.download(tickers=stocks2[5], period='30d', interval='15m', rounding=True)  # "IBM"
    print(stocks2[0])
    print(data1)
    print(stocks2[1])
    print(data2)
    print(stocks2[2])
    print(data3)
    print(stocks2[3])
    print(data4)
    print(stocks2[4]) #remove for 4 assets
    print(data5)  #remove for 4 assets
    print(stocks2[5])  # remove for 4 assets
    print(data6)  # remove for 4 assets
    """"
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data1.index,open = data1['Open'], high=data1['High'], low=data1['Low'], close=data1['Close'], name = 'AAPL'))
    fig.add_trace(go.Candlestick(x=data2.index,open = data2['Open'], high=data2['High'], low=data2['Low'], close=data2['Close'], name = 'GOOG'))
    fig.add_trace(go.Candlestick(x=data3.index,open = data3['Open'], high=data3['High'], low=data3['Low'], close=data3['Close'], name = 'MSFT'))
    fig.add_trace(go.Candlestick(x=data4.index,open = data4['Open'], high=data4['High'], low=data4['Low'], close=data4['Close'], name = 'TSLA'))
    fig.add_trace(go.Candlestick(x=data5.index,open = data5['Open'], high=data5['High'], low=data5['Low'], close=data5['Close'], name = 'AMZN'))

    fig.update_layout(title = 'Share price', yaxis_title = 'Stock Price (USD)')
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
    buttons=list([
    dict(count=15, label='15m', step="minute", stepmode="backward"),
    dict(count=45, label='45m', step= "minute", stepmode="backward"),
    dict(count=1, label='1h', step="hour", stepmode="backward"),
    dict(count=6, label='6h', step="hour", stepmode="backward"),
    dict(step="all")
    ])
    )
    )
    fig.show()
    img=fig.write_image('../dashboard.png')
    """
    #############################
    #############################
    try:
        os.remove('stock1.png')
    except FileNotFoundError:
        pass
    try:
        os.remove('stock2.png')
    except FileNotFoundError:
        pass
    try:
        os.remove('stock3.png')
    except FileNotFoundError:
        pass
    try:
        os.remove('stock4.png')
    except FileNotFoundError:
        pass
    try:                        #5th asset start here
        os.remove('stock5.png')
    except FileNotFoundError:
        pass
    try:                        #6th asset start here
        os.remove('stock6.png')
    except FileNotFoundError:
        pass

    count = 0
    from string import whitespace
    #selection_reversed= np.flip(selection.astype(int))
    #print(selection_reversed)
    quantum_stock = []
    for j in selection:
        if j == 1:
            print('selection:', stocks2[count])
            quantum_stock.append(stocks2[count])
            print(quantum_stock)
            np.savetxt('winners/quantum_stock.csv', [p for p in quantum_stock], delimiter=',', fmt='%s')
            if stocks2[count] == stocks2[0]: #'SEDG'

                msft_data2 = data1
                print(msft_data2)
                a = msft_data2

                a = a.reset_index(drop=False)
                print(a)

                #a = a.rename(columns={'index': 'Datetime'})
                print(a)
                a[['ds', 'y']] = a[['Datetime', 'Adj Close']]
                print(list(a.columns))
                print(a)
                #last = df2[len(df2)-20:]
                #print(last)
                #df2=df2[:20]
                #df2['Datetime'] = pd.to_datetime(df2['Datetime']).dt.tz_localize(None)
                #df2['Datetime'] = pd.to_datetime(df2['Datetime']).dt.date

                #df2=df2.rename(columns={'Datetime':'ds', 'Adj Close':'y'})
                a=a[['ds', 'y']]
                a['ds'] = pd.to_datetime(a['ds']).dt.tz_localize(None)
                a.head()
                print(a)
                #split_date = "2020-10-01"
                #df_train = a.loc[a.ds <= split_date].copy()
                #df_test = a.loc[a.ds > split_date].copy()
                #df2 = df2.reset_index(drop=True)
                #fbp = Prophet(daily_seasonality=False,weekly_seasonality=False,yearly_seasonality=True)
                #fbp.fit(a)
                #forecast = fbp.predict(df_test)
                #forecast.tail()
                #fbp.plot(forecast)
                #future = fbp.make_future_dataframe(periods=60)
                #forecast = fbp.predict(future)
                #plot_plotly(fbp, forecast).show()

                dfb=data1
                print(dfb)
                b = dfb.describe()
                print(b)

                c= dfb.resample('M').mean().head()
                print(c)

                daily_close = dfb[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = dfb.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = dfb['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()
                mav.plot()

                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=dfb.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = dfb['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = dfb['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background") #plt.style.use("default") or "default"
                plt.rcParams['font.size'] = 18 # change font size for all components
                fig = plt.figure()
                plt.title('Quantum Computer recommends investing in '+ stocks2[0], fontsize=30, color='g') #'Algo_trading for SEDG'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad=30
                window = 21
                order = 10
                y_sf = savgol_filter(dfb['Adj Close'], window, order)
                dfb['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,label= 'Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r') #default color 'w' or k
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!" + stocks2[0] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[0] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g') # 'm'
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[0]+ signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[0] + '.csv','a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock1.png')
                plt.close()

                #plt.show()
            if stocks2[count] == stocks2[1]: #'DQ'
                msft_data3 = data2
                #print(msft_data3)
                d = msft_data3.head()
                print(d)
                e = msft_data3.describe()
                print(e)

                f = msft_data3.resample('M').mean().head()
                print(f)

                daily_close = msft_data3[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = msft_data3.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = msft_data3['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()

                mav.plot()


                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=msft_data3.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = msft_data3['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = msft_data3['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background")
                plt.rcParams['font.size'] = 18
                fig = plt.figure()
                plt.title('Quantum Computer recommends investing in '+ stocks2[1], fontsize=30,color='g')#'Algo_trading for DQ'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad=30
                window = 21
                order = 5
                y_sf = savgol_filter(msft_data3['Adj Close'], window, order)
                msft_data3['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,label= 'Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r') #default color 'w' or k
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!"+ stocks2[1] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[1] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g') #or m
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[1] + signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[1] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock2.png')
                plt.close()
                #plt.show()
            if stocks2[count] == stocks2[2]: #'SOL'
                msft_data4 = data3
                #print(msft_data3)
                d = msft_data4.head()
                print(d)
                e = msft_data4.describe()
                print(e)

                f = msft_data4.resample('M').mean().head()
                print(f)

                daily_close = msft_data4[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = msft_data4.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = msft_data4['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()

                mav.plot()


                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=msft_data4.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = msft_data4['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = msft_data4['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background")
                plt.rcParams['font.size'] = 18
                fig = plt.figure()
                plt.title('Quantum Computer recommends investing in '+ stocks2[2], fontsize=30, color='g')#'Algo_trading for SOL'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad=30
                window = 21
                order = 5
                y_sf = savgol_filter(msft_data4['Adj Close'], window, order)
                msft_data4['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,label= 'Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r') #default color 'w' k
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!"+ stocks2[2] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[2] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g') # m
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[2] + signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[2] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock3.png')
                plt.close()
                #plt.show()
            if stocks2[count] == stocks2[3]: #'SPWR'
                msft_data5 = data4
                #print(msft_data3)
                d = msft_data5.head()
                print(d)
                e = msft_data5.describe()
                print(e)

                f = msft_data5.resample('M').mean().head()
                print(f)

                daily_close = msft_data5[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = msft_data5.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = msft_data5['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()

                mav.plot()


                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=msft_data5.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = msft_data5['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = msft_data5['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background")
                plt.rcParams['font.size'] = 18
                fig = plt.figure()
                plt.title('Quantum Computer recommends investing in '+ stocks2[3], fontsize=30, color='g')#'Algo_trading for SPWR'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad=30
                window = 21
                order = 5
                y_sf = savgol_filter(msft_data5['Adj Close'], window, order)
                msft_data5['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,label= 'Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r') #default color 'w'
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!" + stocks2[3] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[3] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g')
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[3] + signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[3] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock4.png')
                plt.close()
                #plt.show()
######################################################## 5th asset starts here
            if stocks2[count] == stocks2[4]: #'SPWR'
                msft_data6 = data5
                #print(msft_data3)
                d = msft_data6.head()
                print(d)
                e = msft_data6.describe()
                print(e)

                f = msft_data6.resample('M').mean().head()
                print(f)

                daily_close = msft_data6[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = msft_data6.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = msft_data6['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()

                mav.plot()


                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=msft_data6.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = msft_data6['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = msft_data6['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background")
                plt.rcParams['font.size'] = 18
                fig = plt.figure()
                plt.title('Quantum Computer recommends investing in '+ stocks2[4], fontsize=30, color='g')#'Algo_trading for SPWR'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad=30
                window = 21
                order = 5
                y_sf = savgol_filter(msft_data6['Adj Close'], window, order)
                msft_data6['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,label= 'Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r') #default color 'w'
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!" + stocks2[4] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[4] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g')
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[4] + signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[4] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock5.png')
                plt.close()
                #plt.show()
########################################################5th asset ends here
######################################################## 6th asset starts here
            if stocks2[count] == stocks2[5]:  # 'SPWR'
                msft_data7 = data6
                # print(msft_data3)
                d = msft_data7.head()
                print(d)
                e = msft_data7.describe()
                print(e)

                f = msft_data7.resample('M').mean().head()
                print(f)

                daily_close = msft_data7[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = msft_data7.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = msft_data7['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()

                mav.plot()

                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=msft_data7.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = msft_data7['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = msft_data7['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background")
                plt.rcParams['font.size'] = 18
                fig = plt.figure()
                plt.title('Quantum Computer recommends investing in ' + stocks2[5], fontsize=30,
                          color='g')  # 'Algo_trading for SPWR'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad = 30
                window = 21
                order = 5
                y_sf = savgol_filter(msft_data7['Adj Close'], window, order)
                msft_data7['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,
                                             label='Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r')  # default color 'w'
                # message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!" + stocks2[5] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[5] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g')
                # message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[5] + signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[5] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock6.png')
                plt.close()
                # plt.show()
            ########################################################6th asset ends here
            else:
                print('All data plotted')

        ######################################
        #For not selected assets
        if j == 0:
            print('selection:', stocks2[count])
            if stocks2[count] == stocks2[0]: #'SEDG'

                msft_data2 = data1
                print(msft_data2)
                a = msft_data2

                a = a.reset_index(drop=False)
                print(a)

                #a = a.rename(columns={'index': 'Datetime'})
                print(a)
                a[['ds', 'y']] = a[['Datetime', 'Adj Close']]
                print(list(a.columns))
                print(a)
                #last = df2[len(df2)-20:]
                #print(last)
                #df2=df2[:20]
                #df2['Datetime'] = pd.to_datetime(df2['Datetime']).dt.tz_localize(None)
                #df2['Datetime'] = pd.to_datetime(df2['Datetime']).dt.date

                #df2=df2.rename(columns={'Datetime':'ds', 'Adj Close':'y'})
                a=a[['ds', 'y']]
                a['ds'] = pd.to_datetime(a['ds']).dt.tz_localize(None)
                a.head()
                print(a)
                #split_date = "2020-10-01"
                #df_train = a.loc[a.ds <= split_date].copy()
                #df_test = a.loc[a.ds > split_date].copy()
                #df2 = df2.reset_index(drop=True)
                #fbp = Prophet(daily_seasonality=False,weekly_seasonality=False,yearly_seasonality=True)
                #fbp.fit(a)
                #forecast = fbp.predict(df_test)
                #forecast.tail()
                #fbp.plot(forecast)
                #future = fbp.make_future_dataframe(periods=60)
                #forecast = fbp.predict(future)
                #plot_plotly(fbp, forecast).show()

                dfb=data1
                print(dfb)
                b = dfb.describe()
                print(b)

                c= dfb.resample('M').mean().head()
                print(c)

                daily_close = dfb[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = dfb.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = dfb['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()
                mav.plot()

                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=dfb.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = dfb['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = dfb['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background") #plt.style.use("default") or "default"
                plt.rcParams['font.size'] = 18 # change font size for all components
                fig = plt.figure()
                plt.title('Avoid investing in'+ stocks2[0], fontsize=30, color='r') #'Algo_trading for SEDG'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad=30
                window = 21
                order = 5
                y_sf = savgol_filter(dfb['Adj Close'], window, order)
                dfb['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,label= 'Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r') #default color 'w' or k
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!" + stocks2[0] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[0] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g') # 'm'
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[0]+ signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[0] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock1.png')
                plt.close()

                #plt.show()
            if stocks2[count] == stocks2[1]: #'DQ'
                msft_data3 = data2
                #print(msft_data3)
                d = msft_data3.head()
                print(d)
                e = msft_data3.describe()
                print(e)

                f = msft_data3.resample('M').mean().head()
                print(f)

                daily_close = msft_data3[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = msft_data3.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = msft_data3['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()

                mav.plot()


                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=msft_data3.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = msft_data3['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = msft_data3['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background")
                plt.rcParams['font.size'] = 18
                fig = plt.figure()
                plt.title('Avoid investing in '+ stocks2[1], fontsize=30, color='r')#'Algo_trading for DQ'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad=30
                window = 21
                order = 5
                y_sf = savgol_filter(msft_data3['Adj Close'], window, order)
                msft_data3['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,label= 'Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r') #default color 'w' or k
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!"+ stocks2[1] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[1] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g') #or m
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[1] + signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[1] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock2.png')
                plt.close()
                #plt.show()
            if stocks2[count] == stocks2[2]: #'SOL'
                msft_data4 = data3
                #print(msft_data3)
                d = msft_data4.head()
                print(d)
                e = msft_data4.describe()
                print(e)

                f = msft_data4.resample('M').mean().head()
                print(f)

                daily_close = msft_data4[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = msft_data4.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = msft_data4['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()

                mav.plot()


                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=msft_data4.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = msft_data4['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = msft_data4['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background")
                plt.rcParams['font.size'] = 18
                fig = plt.figure()
                plt.title('Avoid investing in '+ stocks2[2], fontsize=30, color='r')#'Algo_trading for SOL'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad=30
                window = 21
                order = 5
                y_sf = savgol_filter(msft_data4['Adj Close'], window, order)
                msft_data4['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,label= 'Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r') #default color 'w' k
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!"+ stocks2[2] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[2] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g') # m
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[2] + signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[2] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock3.png')
                plt.close()
                #plt.show()
            if stocks2[count] == stocks2[3]: #'SPWR'
                msft_data5 = data4
                #print(msft_data3)
                d = msft_data5.head()
                print(d)
                e = msft_data5.describe()
                print(e)

                f = msft_data5.resample('M').mean().head()
                print(f)

                daily_close = msft_data5[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = msft_data5.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = msft_data5['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()

                mav.plot()


                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=msft_data5.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = msft_data5['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = msft_data5['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background")
                plt.rcParams['font.size'] = 18
                fig = plt.figure()
                plt.title('Avoid investing in '+ stocks2[3], fontsize=30, color='r')#'Algo_trading for SPWR'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad=30
                window = 21
                order = 5
                y_sf = savgol_filter(msft_data5['Adj Close'], window, order)
                msft_data5['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,label= 'Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r') #default color 'w'
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!" + stocks2[3] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[3] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g')
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[3] + signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[3] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock4.png')
                plt.close()
                #plt.show()

            ######################################################## 5th asset starts here
            if stocks2[count] == stocks2[4]: #'SPWR'
                msft_data6 = data5
                #print(msft_data3)
                d = msft_data6.head()
                print(d)
                e = msft_data6.describe()
                print(e)

                f = msft_data6.resample('M').mean().head()
                print(f)

                daily_close = msft_data6[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = msft_data6.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = msft_data6['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()

                mav.plot()


                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=msft_data6.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = msft_data6['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = msft_data6['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background")
                plt.rcParams['font.size'] = 18
                fig = plt.figure()
                plt.title('Avoid investing in '+ stocks2[4], fontsize=30, color='r')#'Algo_trading for SPWR'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad=30
                window = 21
                order = 5
                y_sf = savgol_filter(msft_data6['Adj Close'], window, order)
                msft_data6['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,label= 'Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r') #default color 'w'
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!" + stocks2[3] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[4] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g')
                #message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[4] + signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[4] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock5.png')
                plt.close()
                #plt.show()
            ######################################################## 5th asset ends here
            ######################################################## 6th asset starts here
            if stocks2[count] == stocks2[5]:  # 'SPWR'
                msft_data7 = data6
                # print(msft_data3)
                d = msft_data7.head()
                print(d)
                e = msft_data7.describe()
                print(e)

                f = msft_data7.resample('M').mean().head()
                print(f)

                daily_close = msft_data7[['Adj Close']]
                daily_return = daily_close.pct_change()
                daily_return.fillna(0, inplace=True)
                print(daily_return)

                mdata = msft_data7.resample('M').apply(lambda x: x[-1])

                monthly_return = mdata.pct_change()

                adj_price = msft_data7['Adj Close']
                mav = adj_price.rolling(window=50).mean()
                adj_price.plot()

                mav.plot()

                short_lb = 50
                long_lb = 120
                signal_df = pd.DataFrame(index=msft_data7.index)
                signal_df['signal'] = 0.0
                signal_df['short_mav'] = msft_data7['Adj Close'].rolling(window=short_lb, min_periods=1,
                                                                         center=False).mean()
                signal_df['long_mav'] = msft_data7['Adj Close'].rolling(window=long_lb, min_periods=1,
                                                                        center=False).mean()
                signal_df['signal'][short_lb:] = np.where(
                    signal_df['short_mav'][short_lb:] > signal_df['long_mav'][short_lb:],
                    1.0, 0.0)
                signal_df['positions'] = signal_df['signal'].diff()
                signal_df[signal_df['positions'] == -1.0]
                plt.style.use("dark_background")
                plt.rcParams['font.size'] = 18
                fig = plt.figure()
                plt.title('Avoid investing in ' + stocks2[5], fontsize=30, color='r')  # 'Algo_trading for SPWR'
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt1 = fig.add_subplot(111, ylabel='Normalised Price')
                plt1.yaxis.labelpad = 30
                window = 21
                order = 5
                y_sf = savgol_filter(msft_data7['Adj Close'], window, order)
                msft_data7['Adj Close'].plot(ax=plt1.twinx().yaxis.set_label_coords(-.5, .5), color='k', lw=2.,
                                             label='Price in $')
                signal_df[['short_mav', 'long_mav']].plot(ax=plt1, lw=2., figsize=(12, 8)).set(yticklabels=[])

                plt1.plot(signal_df.loc[signal_df.positions == -1.0].index,
                          signal_df.short_mav[signal_df.positions == -1.0],
                          'v', markersize=10, color='r')  # default color 'w'
                # message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Sell!" + stocks2[5] + signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["sell", signal_df.loc[signal_df.positions == -1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[5] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt1.plot(signal_df.loc[signal_df.positions == 1.0].index,
                          signal_df.short_mav[signal_df.positions == 1.0],
                          '^',
                          markersize=10, color='g')
                # message = client.messages.create(to="+31619809790", from_="+19707030969",
                #                                 body="Buy!"+ stocks2[5] + signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all())
                ##-----------
                row = ''.join(["buy", signal_df.loc[signal_df.positions == 1.0].index.strftime("%Y/%m/%d").all()])
                with open(stocks2[5] + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    # time.sleep(60)
                    writer.writerow(row)
                    print(row)
                ##-----------
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.rcParams.update({'font.size': 18})

                plt.savefig('stock6.png')
                plt.close()
                # plt.show()
            ######################################################## 6th asset ends here

            else:
                print('All data plotted')
        count = count + 1
        ######################################

Builder.load_string("""
<User>:
    username:username
    user_label:user_label
    but_1:but_1
    asset:asset
    asset_label:asset_label
    but_2:but_2
    # cols: 2
    Label:
        id: user_label
        font_size: 30
        color: 0.6, 0.6, 0.6, 1
        text_size: self.width, None
        halign: 'left'
        text:'Enter start date below'

    TextInput:
        id: username
        font_size: 30
        pos_hint:{"x":0.3,"y":0.25}
        size_hint: 0.4, 0.07
        color: 0.6, 0.6, 0.6, 1
        text_size: self.width, None
        halign: 'left'
        
    Label:
        id: asset_label
        font_size: 30
        color: 0.6, 0.6, 0.6, 1
        text_size: self.width, None
        halign: 'right'
        text:'Enter assets above'

    TextInput:
        id: asset
        font_size: 30
        pos_hint:{"x":0.3,"y":0.7}
        size_hint: 0.4, 0.07
        color: 0.6, 0.6, 0.6, 1
        text_size: self.width, None
        halign: 'right'
        
        
    Button:
        id: but_1
        font_size: 20
        pos_hint:{"x":0.3,"y":0.15}
        size_hint: 0.4, 0.07
        text: 'Save start date'
        on_press:
            root.save_username()
            root.set_username()
            root.save_asset()
            root.set_asset()
            root.manager.current = 'get_user'
            root.manager.current = 'get_asset'
            

    Button:
        id: but_2
        font_size: 20
        pos_hint:{"x":0.3,"y":0.60}
        size_hint: 0.4, 0.07
        text: 'Save assets'
        on_press:
            root.save_username()
            root.set_username()
            root.save_asset()
            root.set_asset()
            
<GetUser>:
    load_username:load_username
    user_label:user_label
    but_1:but_1
    load_asset:load_asset
    asset_label:asset_label
    but_2:but_2
    
    # cols: 2
    Label:
        id: user_label
        font_size: 30
        color: 0.6, 0.6, 0.6, 1
        text_size: self.width, None
        halign: 'left'
        text:'Received startdate'

    TextInput:
        id: load_username
        font_size: 30
        pos_hint:{"x":0.3,"y":0.25}
        size_hint: 0.4, 0.07
        color: 0.6, 0.6, 0.6, 1
        text_size: self.width, None
        halign: 'center'
        disabled:False
    Button:
        id: but_1
        font_size: 20
        pos_hint:{"x":0.3,"y":0.15}
        size_hint: 0.4, 0.07
        text: 'save start date'
        on_press:
            
            root.manager.current = 'user'
    
      
    Label:
        id: asset_label
        font_size: 30
        color: 0.6, 0.6, 0.6, 1
        text_size: self.width, None
        halign: 'right'
        text:'Received assets from previous page'

    TextInput:
        id: load_asset
        font_size: 30
        pos_hint:{"x":0.3,"y":0.7}
        size_hint: 0.4, 0.07
        color: 0.6, 0.6, 0.6, 1
        text_size: self.width, None
        halign: 'right'
        disabled:False
    Button:
        id: but_2
        font_size: 20
        pos_hint:{"x":0.3,"y":0.60}
        size_hint: 0.4, 0.07
        text: 'save assets'
        on_press:
            root.manager.current = 'asset'
            
<CloseApp>: 

    Label:
        text: 'Popup text'
        size_hint: .4, .15
        pos_hint:{'center_x': .5, 'center_y': .7}
        halign: "center"
        valign: "center"      
    Button: 
        id: but_3
        font_size: 20
        pos_hint:{"x":0.3,"y":0.60}
        size_hint: 0.4, 0.07
        text: 'OK'
        on_release: 
            app.root.current_screen.close()                 
""")


from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, NumericProperty, StringProperty


class User(Screen,Widget):
    last_name_text_input = ObjectProperty()
    ego = NumericProperty(0)
    name = StringProperty('')

    def save_asset(self):
        global bbbb
        bbbb = self.asset.text
        print('asset:',bbbb)
        return bbbb

    def save_username(self):
        global aaaa
        aaaa = self.username.text
        print('saved:',aaaa)
        return aaaa

    def save(self):
        with open("username.txt", "w") as fobj:
            fobj.write(str(self.username))
        return fobj

    def load(self):
        with open("username.txt") as fobj:
            for username in fobj:
                self.username = username.rstrip()
        return  fobj

    def submit_username(self):
        self.username = self.username.text
        print("Assign username: {}".format(self.username))
        self.save()
        self.username = ''
        print("Reset username: {}".format(self.username))
        self.load()
        print("Loaded username: {}".format(self.username))


    def set_username(self):  # <--- Asign the name here
        screens = App.get_running_app().root.screens
        other_screen = None
        text = ""
        for screen in screens:
            if screen.name == "user":
                text = screen.username.text
            elif screen.name == "get_user":
                other_screen = screen

        other_screen.load_username.text = text
        return text

    def set_asset(self):  # <--- Asign the name here
        screens = App.get_running_app().root.screens
        other_screen = None
        text = ""
        for screen in screens:
            if screen.name == "asset":
                text = screen.asset.text
            elif screen.name == "get_asset":
                other_screen = screen

        other_screen.load_asset.text = text
        return text

class GetUser(Screen):

        pass


sm = ScreenManager()
sm.add_widget(User(name='user'))
sm.add_widget(GetUser(name='get_user'))
sm.add_widget(User(name='saved:'))
sm.add_widget(User(name='asset:'))
sm.add_widget(User(name='asset'))
sm.add_widget(GetUser(name='get_asset'))

class TradeDate(App):

    def build(self):

        return sm

class CloseApp(App):
   
    def build(self):

        Window.bind(on_request_close=self.on_request_close)
        return Label(text='Please Close App')

    def on_request_close(self, *args):
        self.textpopup(title='Exit', text='Are you sure?')
        return True

    def textpopup(self, title='', text=''):
        #Open the pop-up with the name.

        #:param title: title of the pop-up to open
        #:type title: str
        #:param text: main text of the pop-up to open
        #:type text: str
        #:rtype: None
        
        box = BoxLayout(orientation='vertical')
        box.add_widget(Label(text=text))
        mybutton = Button(text='OK', size_hint=(1, 0.25))
        box.add_widget(mybutton)
        popup = Popup(title=title, content=box, size_hint=(None, None), size=(600, 300))
        mybutton.bind(on_release=lambda b: self.stop()) #on_release=self.stop()
        #popup.open()

        return box

    async def setUp(self):
        """ Bring server up. """
        app = FastAPI()
        self.proc = Process(target=uvicorn.run,
                            args=(app.api,),
                            kwargs={
                                "host": "127.0.0.1",
                                "port": 8000,
                                "log_level": "info"},
                            daemon=True)
        self.proc.start()
        await asyncio.sleep(0.1)  # time for the server to start

class MainApp(): #MainApp(App)

   async def run(self): # build(self)
        result = await api.asset(Request)

        print(result)
        global startdate
        startdate = result[1]
        assets = result[0]
        print(startdate) # aaaa

        print(assets) # bbbb
        print(int(list(map(str, startdate.split('-')))[1])) #aaaa
        gen_cov(startdate) #aaaa
        plotstock('yahoo')
        gen_algotrade()



"""
        layout = FloatLayout(size=(200, 200))

        k=[0,1,2,3]
        count = 0
        optimalSelection=[]
        for j in selection:
            if j==1:
                print('selection:',stocks2[count])
                optimalSelection.append(stocks2[count])
                if stocks2[count] == stocks2[0]:
                    img3 = Image(source='../stock1.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .3, 'center_y': .25})
                    layout.add_widget(img3)
                elif stocks2[count] == stocks2[1]:
                    img4 = Image(source='../stock2.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
                elif stocks2[count] == stocks2[2]:
                    img4 = Image(source='../stock3.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
                elif stocks2[count] == stocks2[2]:
                    img4 = Image(source='../stock3.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
                if stocks2[count] == stocks2[1]:
                    img3 = Image(source='../stock2.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .3, 'center_y': .25})
                    layout.add_widget(img3)
                elif stocks2[count] == stocks2[0]:
                    img4 = Image(source='../stock1.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
                elif stocks2[count] == stocks2[2]:
                    img4 = Image(source='../stock3.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
                elif stocks2[count] == stocks2[3]:
                    img4 = Image(source='../stock4.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
                if stocks2[count] == stocks2[2]:
                    img3 = Image(source='../stock3.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .3, 'center_y': .25})
                    layout.add_widget(img3)
                elif stocks2[count] == stocks2[0]:
                    img4 = Image(source='../stock1.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
                elif stocks2[count] == stocks2[1]:
                    img4 = Image(source='../stock2.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
                elif stocks2[count] == stocks2[3]:
                    img4 = Image(source='../stock4.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
                if stocks2[count] == stocks2[3]:
                    img3 = Image(source='../stock4.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .3, 'center_y': .25})
                    layout.add_widget(img3)
                elif stocks2[count] == stocks2[0]:
                    img4 = Image(source='../stock1.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
                elif stocks2[count] == stocks2[1]:
                    img4 = Image(source='../stock2.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
                elif stocks2[count] == stocks2[2]:
                    img4 = Image(source='../stock3.png',
                                 size_hint=(0.7, 0.5),
                                 pos_hint={'center_x': .7, 'center_y': .25})
                    layout.add_widget(img4)
            count =count+1

        self.startdate = TextInput(text= format(optimalSelection), size_hint=(.2, .2), pos=(50, 20))
        self.enddate = TextInput(text= format(value), size_hint=(0.2, .2), pos=(600, 20))
        img1 = Image(source='../covariance.png',
                     size_hint=(0.8, 0.5),
                     pos_hint={'center_x': .3, 'center_y': .75})
        img2 = Image(source='../stock.png',
                     size_hint=(0.8, 0.5),
                     pos_hint={'center_x': .7, 'center_y': .75})

        #img3 = Image(source='../stock1.png',
        #             size_hint=(0.7, 0.5),
        #             pos_hint={'center_x': .3, 'center_y': .25})
        #img4 = Image(source='../DQ.png',
        #             size_hint=(0.7, 0.5),
        #             pos_hint={'center_x': .7, 'center_y': .25})
        layout.add_widget(img1)
        layout.add_widget(img2)
        #layout.add_widget(img3)
        #layout.add_widget(img4)
        button = Button(text='Submit', size_hint=(.1, .1), pos=(350,20))

        #layout.add_widget(self.startdate)
        #layout.add_widget(self.enddate)
        #layout.export_to_png("covariance.png")
        return layout
"""
from io import StringIO
import sys
from kivy.storage.jsonstore import JsonStore

if __name__ == '__main__':

    #TradeDate().run()
    #print(sm)
    app = MainApp()
    app.run()
    #plotstock('yahoo')

class MainWindow(BoxLayout):

    # We create a dictionary of all our possible methods to call, along with keys
    def command_dict(self):
        return {
            'one': self.command_one,
            'two': self.command_two,
            'three': self.command_three
        }

    def process_command(self):
        # We grab the text from the user text input as a key
        command_key = self.ids.fetch_key_and_process_command.text

        # We then use that key in the command's built in 'get_method' because it is a dict
        # then we store it into a variable for later use
        called_command = self.command_dict().get(command_key, 'default')

        try:
            # The variable is a method, so by adding we can call it by simple adding your typical () to the end of it.
            called_command()

        except TypeError:
            # However we use an exception clause to catch in case people enter a key that doesn't exist
            self.ids.fetch_key_and_process_command.text = 'Sorry, there is no command key: ' + command_key

    # These are the three commands we call from our command dict.

    def command_one(self):

        self.ids.fetch_key_and_process_command.text = 'Command One has Been Processed'

    def command_two(self):
        self.ids.fetch_key_and_process_command.text = 'Command Two has Been Processed'

    def command_three(self):
        self.ids.fetch_key_and_process_command.text = 'Command Three has been Processed'


import sqlite3


# Function for Convert Binary Data
# to Human Readable Format
def convertToBinaryData(filename):
    # Convert binary format to images
    # or files data
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


def insertBLOB(name, photo):
    try:

        # Using connect method for establishing
        # a connection
        sqliteConnection = sqlite3.connect('app.db')
        cursor = sqliteConnection.cursor()
        print("Connected to SQLite")

        # insert query
        sqlite_insert_blob_query = """INSERT INTO stock_image
								(stock_id, strategy_image) VALUES (?, ?)"""

        # Converting human readable file into
        # binary data
        empPhoto = convertToBinaryData(photo)

        # Convert data into tuple format
        data_tuple = (name, empPhoto)

        # using cursor object executing our query
        cursor.execute(sqlite_insert_blob_query, data_tuple)
        sqliteConnection.commit()
        print("Image and file inserted successfully as a BLOB into a table")
        cursor.close()

    except sqlite3.Error as error:
        print("Failed to insert blob data into sqlite table", error)

    finally:
        if sqliteConnection:
            sqliteConnection.close()
            print("the sqlite connection is closed")


#insertBLOB("SEDQ", "C:/Users/User/qd2d/SEDG.png")
#insertBLOB("DQ", "C:/Users/User/qd2d/DQ.png")

"""
from os import curdir
from os.path import join as pjoin

from http.server import BaseHTTPRequestHandler, HTTPServer

class StoreHandler(BaseHTTPRequestHandler):
    store_path = pjoin(curdir, 'covariance.jpg')

    def do_GET(self):
        if self.path == '/covariance.jpg':
            with open(self.store_path) as fh:
                self.send_response(200)
                self.send_header('Content-type', 'text/jpg')
                self.end_headers()
                self.wfile.write(fh.read().encode())

    def do_POST(self):
        if self.path == '/covariance.jpg':
            length = self.headers['content-length']
            data = self.rfile.read(int(length))

            with open(self.store_path, 'w') as fh:
                fh.write(data.decode())

            self.send_response(200)


server = HTTPServer(('localhost', 8080), StoreHandler)
server.serve_forever()



import random
import socket, select
from time import gmtime, strftime
from random import randint

imgcounter = 1
basename = "SEDG.png"

HOST = '127.0.1.1'
PORT = 8080

connected_clients_sockets = []

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(10)

connected_clients_sockets.append(server_socket)

while True:

    read_sockets, write_sockets, error_sockets = select.select(connected_clients_sockets, [], [])

    for sock in read_sockets:

        if sock == server_socket:

            sockfd, client_address = server_socket.accept()
            connected_clients_sockets.append(sockfd)

        else:
            try:

                data = sock.recv(4096)
                txt = str(data)

                if data:

                    if data.startswith('SIZE'):
                        tmp = txt.split()
                        size = int(tmp[1])

                        print('got size')

                        sock.sendall("GOT SIZE")

                    elif data.startswith('BYE'):
                        sock.shutdown()

                    else :

                        myfile = open(basename % imgcounter, 'wb')
                        myfile.write(data)

                        data = sock.recv(40960000)
                        if not data:
                            myfile.close()
                            break
                        myfile.write(data)
                        myfile.close()

                        sock.sendall("GOT IMAGE")
                        sock.shutdown()
            except:
                sock.close()
                connected_clients_sockets.remove(sock)
                continue
        imgcounter += 1
server_socket.close()
"""