# Import libraries
import sqlite3
import json
import numpy as np
import pandas as pd
import parser_libraries
import lxml
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
import matplotlib.pyplot as plt
import urllib3
import nltk
nltk.download
#from urllib3.request import RequestMethods
#from urllib3 import request # urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

#Calling from database the symbols we want to put in our portfolio for optimization
cursor.execute("""
    SELECT symbol, company FROM stock
    join stock_strategy on stock_strategy.stock_id= stock.id
    where stock_strategy.strategy_id = ?
""",(strategy_id,))

stocks =cursor.fetchall()

symbols = [stock['symbol'] for stock in stocks]

connection.commit()


http = urllib3.PoolManager()
# Parameters
n = 10  # the # of article headlines displayed per ticker iso n=3

#Calling from sqlite database the symbols we want to put in our portfolio for optimization

#tickers = symbols#['SEDG','SPWR','SOL','DQ','TSLA','AAPL'] #['GM','UNG', 'NKLA', 'IBM', 'AAPL', 'AMZN', 'GOOG', 'MSFT', 'TSLA', 'NIO','LI', 'BOIL']

#Calling from database.json the symbols we want to put in our portfolio for optimization
with open('database.json') as f:
    symbols_data = json.load(f)

tickers = [symbols_data['1'], symbols_data['2'],symbols_data['3'],symbols_data['4'],symbols_data['5'], symbols_data['6']]

# Get Data
finwiz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

for ticker in tickers:
    pd.set_option('display.max_colwidth', 25)

    # Input
    symbol = ticker
    print('Getting data for ' + symbol + '...\n')

    # Set up scraper
    url = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    html = soup(webpage, "html.parser")
    #html = BeautifulSoup(resp, 'lxml')
    news_table = html.find(id='news-table')
    print(news_table)
    news_tables[ticker] = news_table

try:
    for ticker in tickers:
        df = news_tables[ticker]
        print(df)
        df_tr = df.findAll('tr')

        print('\n')
        print('Recent News Headlines for {}: '.format(ticker))

        for i, table_row in enumerate(df_tr):
            a_text = table_row.a.text
            td_text = table_row.td.text
            td_text = td_text.strip()
            print(a_text, '(', td_text, ')')
            if i == n - 1:
                break
except KeyError:
    pass

# Iterate through the news
parsed_news = []
for file_name, news_table in news_tables.items():
    for x in news_table.findAll('tr'):
        text = x.a.get_text()
        date_scrape = x.td.text.split()

        if len(date_scrape) == 1:
            time = date_scrape[0]

        else:
            date = date_scrape[0]
            time = date_scrape[1]

        ticker = file_name.split('_')[0]

        parsed_news.append([ticker, date, time, text])

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

columns = ['Ticker', 'Date', 'Time', 'Headline']
news = pd.DataFrame(parsed_news, columns=columns)
scores = news['Headline'].apply(analyzer.polarity_scores).tolist() #polarity_scores

df_scores = pd.DataFrame(scores)
news = news.join(df_scores, rsuffix='_right')
print(news)
# View Data

for new in news:
    print(news['Date'][2])
    if news['Date'][2] != "Today" and news['Date'][2] != "Apr-19-24":
        news['Date'] = pd.to_datetime(news.Date).dt.date  # Added for loop and if statement otherwise this line alone works
    else:
        pass
unique_ticker = news['Ticker'].unique().tolist()
news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

values = []
for ticker in tickers:
    dataframe = news_dict[ticker]
    dataframe = dataframe.set_index('Ticker')
    dataframe = dataframe.drop(columns=['Headline'])
    print('\n')
    print(dataframe.head())

    mean = round(dataframe['compound'].mean(), 2)
    values.append(mean)

df = pd.DataFrame(list(zip(tickers, values)), columns=['Ticker', 'Mean Sentiment'])
#df = df.set_index('Ticker')
#df = df.sort_values('Mean Sentiment', ascending=False)
#print('\n')
print(df)

# creation of a pie chart of maximum sharpe allocation
#########################################################

# Creating dataset
dataframe = df

print(dataframe)

cars = tickers


# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100. * np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)

"""
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title("Machine Learning News based Sentiment Analysis")
ax.legend(cars,
          title="Assets",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
"""
plt.style.use("dark_background")
# Figure Size


ax=dataframe.groupby(['Ticker']).sum().plot(
    kind='bar', y='Mean Sentiment', color=['blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue','blue','blue','blue','blue'])
# Add annotation to bars
"""
for i in ax.patches:
    plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
             str(round((i.get_width()), 2)),
             fontsize=10, fontweight='bold',
             color='white')
"""

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
plt.savefig('stock10.png')
plt.close()
