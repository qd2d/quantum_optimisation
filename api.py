from django import views
from django.conf.urls import url
from fastapi.responses import StreamingResponse
from flask import Flask, render_template
from pip._vendor.distlib.compat import raw_input
from threading import Thread
from time import sleep

import redis
import json
from typing import List

from fastapi import FastAPI, File, UploadFile
from flask import Flask, request, Response #added request for authentication
from fastapi.responses import HTMLResponse
import random
import imgbbpy


from kivy.app import App
import yarl
import asyncio
import sqlite3, config
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import datetime
#import stock_data


from fastapi import Depends, HTTPException, status, Path
from fastapi.security import HTTPBearer, OAuth2PasswordBearer
import multipart
from slugify import slugify
import html
from os.path import isfile
import sys
import subprocess
import aioredis
import aiohttp
import uvicorn

from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter



####START OF SECURITY BLOCK

import os
generate_key = print(os.urandom(24).hex())

SECRET = 'your secret'

import secrets
from typing import Dict
from fastapi import FastAPI, HTTPException, status, Depends, Cookie, Form, Response, Request
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from hashlib import sha256

app = FastAPI()
security = HTTPBasic()
app.secret_key = "your key"
app.sessions= []

###################
from base64 import b64encode

from fastapi import FastAPI, Security
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.testclient import TestClient
from requests.auth import HTTPBasicAuth

#app = FastAPI()

security = HTTPBasic(realm="simple")


@app.get("/login")
def read_current_user(response: Response, credentials: HTTPBasicCredentials = Security(security)):
    correct_username = secrets.compare_digest(credentials.username, "your username")
    correct_password = secrets.compare_digest(credentials.password, "your password")
    if (correct_username and correct_password):
        session_token = sha256(bytes(f"{credentials.username}{credentials.password}{app.secret_key}", encoding='utf8')).hexdigest()
        response.status_code = status.HTTP_302_FOUND
        response.set_cookie(key="session_token", value=session_token)
        app.sessions += session_token
        response.headers['Location'] = "/strategy/19"
        return response
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect login or password",
        )
    #return {"username": credentials.username, "password": credentials.password}


client = TestClient(app)

openapi_schema = {
    "openapi": "3.0.2",
    "info": {"title": "FastAPI", "version": "0.1.0"},
    "paths": {
        "/users/me": {
            "get": {
                "responses": {
                    "200": {
                        "description": "Successful Response",
                        "content": {"application/json": {"schema": {}}},
                    }
                },
                "summary": "Read Current User",
                "operationId": "read_current_user_users_me_get",
                "security": [{"HTTPBasic": []}],
            }
        }
    },
    "components": {
        "securitySchemes": {"HTTPBasic": {"type": "http", "scheme": "basic"}}
    },
}


def test_openapi_schema():
    response = client.get("/openapi.json")
    assert response.status_code == 200, response.text
    assert response.json() == openapi_schema


def test_security_http_basic():
    auth = HTTPBasicAuth(username="john", password="secret")
    response = client.get("/strategy/19", auth=auth)
    assert response.status_code == 200, response.text
    assert response.json() == {"username": "your username", "password": "your password"}


def test_security_http_basic_no_credentials():
    response = client.get("/strategy/19")
    assert response.json() == {"detail": "Not authenticated"}
    assert response.status_code == 401, response.text
    assert response.headers["WWW-Authenticate"] == 'Basic realm="simple"'


def test_security_http_basic_invalid_credentials():
    response = client.get(
        "/strategy/19", headers={"Authorization": "Basic notabase64token"}
    )
    assert response.status_code == 401, response.text
    assert response.headers["WWW-Authenticate"] == 'Basic realm="simple"'
    assert response.json() == {"detail": "Invalid authentication credentials"}


def test_security_http_basic_non_basic_credentials():
    payload = b64encode(b"johnsecret").decode("ascii")
    auth_header = f"Basic {payload}"
    response = client.get("/strategy/19", headers={"Authorization": auth_header})
    assert response.status_code == 401, response.text
    assert response.headers["WWW-Authenticate"] == 'Basic realm="simple"'
    assert response.json() == {"detail": "Invalid authentication credentials"}


#######################List of Symbols###############################

@app.get("/symbols")
async def root(request: Request):
    with open('database.json') as f:
        data = json.load(f)
    return templates.TemplateResponse("strategy.html", {"request":request,"tododict":data})

@app.get("/delete/{id}")
async def delete_todo(request: Request, id: str):
    with open('database.json') as f:
        data = json.load(f)
    del data[id]
    with open('database.json','w') as f:
        json.dump(data,f)
    return RedirectResponse("/strategy/19", 303)

@app.post("/add")
async def add_todo(request: Request):
    with open('database.json') as f:
        data = json.load(f)
    formdata = await request.form()
    newdata = {}
    i=1
    for id in data:
        newdata[str(i)] = data[id]  #newdata[str(i)] = data[id]
        i+=1
    newdata[str(i)] = formdata["newtodo"] #newdata[str(i)]
    print(newdata)
    with open('database.json','w') as f:
        json.dump(newdata,f)
    return RedirectResponse("/strategy/19", 303)
################end of symbols list block########################

##################
from functools import wraps
from flask import request, Response
import threading
def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return username == 'your username' and password == 'your password'

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})
app2 = Flask(__name__)
import _thread
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        def handle_sub_view(req):
            with app2.test_request_context():
                from flask import request

                request = req
                auth = request.authorization
                if not auth or not check_auth(auth.username, auth.password):
                    return authenticate()
                return f(*args, **kwargs)

        _thread.start_new_thread(handle_sub_view, (request,))
        return HTMLResponse('<a href="https://xxxxxxx.xxxxx.xx/strategy/19">login</a>')
    return decorated

@app.get("/strategy/{strategy_id}")

def strategy(request: Request, strategy_id):

    connection = sqlite3.connect("app.db")
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, name
        FROM strategy
        WHERE id = ?
    """, (strategy_id,))

    strategy = cursor.fetchone()

    cursor.execute("""
        SELECT symbol, company
        FROM stock JOIN stock_strategy on stock_strategy.stock_id = stock.id
        WHERE strategy_id = ?
    """, (strategy_id,))

    stocks = cursor.fetchall()
    with open('database.json') as f:
        data = json.load(f)
    return templates.TemplateResponse("strategy.html",
                                      {"request": request, "stocks": stocks, "strategy": strategy, "tododict":data})
    connection.commit()

##################
@app.get("/strategy/19")
@requires_auth
def strategy(request: Request, strategy_id):
    connection = sqlite3.connect("app.db")
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT id, name
        FROM strategy
        WHERE id = ?
    """, (strategy_id,))

    strategy = cursor.fetchone()

    cursor.execute("""
        SELECT symbol, company
        FROM stock JOIN stock_strategy on stock_strategy.stock_id = stock.id
        WHERE strategy_id = ?
    """, (strategy_id,))

    stocks = cursor.fetchall()
    return templates.TemplateResponse("strategy.html",
                                      {"request": request, "stocks": stocks, "strategy": strategy})
    connection.commit()


"""
@app.get("/login")
def Login(response: Response,credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "root")
    correct_password = secrets.compare_digest(credentials.password, "toor")
    if (correct_username and correct_password):
        session_token = sha256(bytes(f"{credentials.username}{credentials.password}{app.secret_key}", encoding='utf8')).hexdigest()
        response.status_code = status.HTTP_302_FOUND
        response.set_cookie(key="session_token", value=session_token)
        app.sessions += session_token
        response.headers['Location'] = "/strategy/19"
        return response
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect login or password",
        )
"""
####END OF SECURITY BLOCK

token_auth_scheme = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")



origins = [
    "https://localhost",
    "https://localhost:8000"
    "https://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



templates=Jinja2Templates(directory="../xxxx")

@app.get("/items/")
async def read_items(token: str = Depends(oauth2_scheme)):
    return {"token": token}


@app.get("/")
def index(request: Request):
    print(request)
    stock_filter=request.query_params.get('filter', True)
    connection = sqlite3.connect("app.db")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    if stock_filter == 'new_closing_highs':
        cursor.execute("""
           SELECT * FROM
                ( SELECT symbol, company, stock_id, MAX(close) AS max_close, date FROM stock_price
                LEFT JOIN stock ON stock.id = stock_price.stock_id
                GROUP BY stock_id
                ORDER BY symbol
           ) WHERE date = (SELECT MAX(date) FROM stock_price);
           """)
    else:
        cursor.execute(""" 
            SELECT id, symbol, company FROM stock ORDER BY symbol
        """)
    rows=cursor.fetchall()
    return templates.TemplateResponse("index.html", {"request": request, "stocks":rows})
    connection.commit()

@app.get("/news")
def news(request: Request):
    print(request)
    stock_filter=request.query_params.get('filter', True)
    connection = sqlite3.connect("app.db")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    if stock_filter == 'new_closing_highs':
        cursor.execute("""
           SELECT * FROM
                ( SELECT symbol, company, stock_id, MAX(close) AS max_close, date FROM stock_price
                LEFT JOIN stock ON stock.id = stock_price.stock_id
                GROUP BY stock_id
                ORDER BY symbol
           ) WHERE date = (SELECT MAX(date) FROM stock_price);
           """)
    else:
        cursor.execute(""" 
            SELECT id, symbol, company FROM stock ORDER BY symbol
        """)
    rows=cursor.fetchall()
    return templates.TemplateResponse("news.html", {"request": request, "stocks":rows})
    connection.commit()

@app.get("/stock/{symbol}")
def stock_detail(request: Request, symbol):
    connection = sqlite3.connect("app.db")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute("""
        SELECT * FROM strategy
    """)

    strategies = cursor.fetchall()
    cursor.execute(""" 
        SELECT id, symbol, company FROM stock WHERE symbol = ?
    """, (symbol,))
    row = cursor.fetchone()
    #return templates.TemplateResponse("stock_detail.html", {"request": request, "stock": row})

    cursor.execute(""" 
        SELECT * FROM stock_price WHERE stock_id = ?
    """, (row['id'],))
    bars = cursor.fetchall()

    return templates.TemplateResponse("stock_detail.html", {"request": request, "stock": row,  "bars": bars, "strategies": strategies})
    connection.commit()

@app.post("/apply_strategy")
def apply_strategy(strategy_id: int=Form(...), stock_id: int=Form(...)):
    connection = sqlite3.connect("app.db")
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO stock_strategy (stock_id, strategy_id) VALUES (?, ?)
    """, (stock_id,strategy_id))

    connection.commit()

    return RedirectResponse(url=f"/strategy/{strategy_id}",status_code=303)
"""

@app.get("/strategy/{strategy_id}")
def read_current_user(response: Response, credentials: HTTPBasicCredentials = Depends(security)):

    correct_username = secrets.compare_digest(credentials.username, "root")
    correct_password = secrets.compare_digest(credentials.password, "toor")
    if (correct_username and correct_password):
        return correct_username
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect login or password",
        )

"""

"""
@app.get("/login")
def Login(response: Response,credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "root")
    correct_password = secrets.compare_digest(credentials.password, "toor")
    if (correct_username and correct_password):
        session_token = sha256(bytes(f"{credentials.username}{credentials.password}{app.secret_key}", encoding='utf8')).hexdigest()
        response.status_code = status.HTTP_302_FOUND
        response.set_cookie(key="session_token", value=session_token)
        app.sessions += session_token
        response.headers['Location'] = "/strategy/19"
        return response

    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect login or password",
        )
"""


@app.get('/refreshdatabase')
async def refreshdatabase(): # token: str = Depends(oauth2_scheme)
    #async with aiohttp.ClientSession(trust_env=True) as session:
        #async with session.get(url=yarl.URL("https://127.0.1.1:8081"), ssl=False) as response:
            #assert response.status == 200
    connection = sqlite3.connect("app.db")
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    sql = 'DELETE FROM stock_strategy WHERE strategy_id=?'

    cursor.execute(sql, (19,))
    connection.commit()
    return "Database refreshed successfully"



"""
@app.get("/login")
async def login(request: Request):
    async with aiohttp.ClientSession(trust_env=True) as session:
        async with session.get(url=yarl.URL("https://127.0.1.1:8081"), ssl=False) as response:
            assert response.status == 200
            result1 = "Type a number"
            return templates.TemplateResponse('strategy.html', context={'request': request, 'result': result1})
"""

@app.post("/asset")
async def asset(request: Request, assets: str = Form(...)):
    #async with aiohttp.ClientSession(trust_env=True) as session:
        #async with session.get(url=yarl.URL("https://127.0.1.1:8081"), ssl=False) as response:
            #assert response.status == 200
    #global startdate
    with open('database.json') as f:
        symbols_data = json.load(f)
    startdate = symbols_data['7'] #"2022/8/1"

    result2 = assets

    return assets, startdate
    #return "Assets:", assets, "Start date:",startdate,"Optimisation succesfull!"


""""
            return templates.TemplateResponse('strategy.html', context={'request': request, 'result': result2})
"""


@app.post("/start")
async def start(request: Request, startdate: str = Form(...)):
    #async with aiohttp.ClientSession(trust_env=True) as session:
        #async with session.get(url=yarl.URL("https://127.0.1.1:8081"), ssl=False) as response:
            #assert response.status == 200
    global result3
    result3 = startdate
    print(result3)
    import stock_data
    # result = stock_data.TradeDate().run()

    await stock_data.MainApp().run()

    return startdate
"""
            return templates.TemplateResponse('strategy.html', context={'request': request, 'result': result3})
"""



"""
@app2.route('/login', methods=['POST'])
def login(request):
    if request.method == 'POST':
        assets = request.POST.get('assets')
        startdate = request.POST.get('startdate')
        return assets, startdate
    else:
        assets  = 'IBM/GOOG/MSFT/TSLA'
        startdate = '2021/1/1'

login(Request)
"""

from django.shortcuts import redirect, render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse




@app.get('/optimization')
async def optimization(): # token: str = Depends(oauth2_scheme)
    #async with aiohttp.ClientSession(trust_env=True) as session:
        #async with session.get(url=yarl.URL("https://127.0.1.1:8081"), ssl=False) as response:
            #assert response.status == 200

    import time


    from importlib import reload


    reload(time)
    # measure process time
    t0 = time.clock()
    import classical_stock_data
    reload(classical_stock_data)
    #import sentiment_analytics
    #reload(sentiment_analytics)
    import stock_data
    reload(stock_data)
    #import stoch
    #reload(stoch)
    #result = stock_data.TradeDate().run()

    time.clock()-t0, "seconds process time"
    await stock_data.MainApp().run()
    from psutil import process_iter
    from signal import SIGTERM  # or SIGKILL


    for proc in process_iter():
        for conns in proc.connections(kind='inet'):
            if conns.laddr.port == 8001:
                proc.send_signal(SIGTERM)  # or SIGKILL
                continue
            else:
                pass
    from subprocess import Popen
    p = Popen("openpage.bat", cwd=r"C:\\Users\User\xxxx")
    #stdout, stderr = p.communicate()

    import webbrowser

    webbrowser.get(using=None)
    return (webbrowser.open('https://xxxxx.xxxx.xx/strategy/19'))

#####################TECHNICAL ANALYSIS###############################################
@app.get('/technicalanalysis')
async def technicalanalysis(): # token: str = Depends(oauth2_scheme)
    #async with aiohttp.ClientSession(trust_env=True) as session:
        #async with session.get(url=yarl.URL("https://127.0.1.1:8081"), ssl=False) as response:
            #assert response.status == 200

    import time


    from importlib import reload


    reload(time)
    # measure process time
    t0 = time.clock()
    #import classical_stock_data
    #reload(classical_stock_data)
    #import sentiment_analytics
    #reload(sentiment_analytics)
    #import stock_data
    #reload(stock_data)
    import stoch
    reload(stoch)
    #result = stock_data.TradeDate().run()

    time.clock()-t0, "seconds process time"
    #await stoch#stock_data.MainApp().run()
    from psutil import process_iter
    from signal import SIGTERM  # or SIGKILL


    for proc in process_iter():
        for conns in proc.connections(kind='inet'):
            if conns.laddr.port == 8001:
                proc.send_signal(SIGTERM)  # or SIGKILL
                continue
            else:
                pass
    from subprocess import Popen
    p = Popen("openpage.bat", cwd=r"C:\\Users\User\xxxx")
    #stdout, stderr = p.communicate()

    import webbrowser

    webbrowser.get(using=None)
    return (webbrowser.open('https://xxxxxx.xxxxx.xx/strategy/19'))

 ####################QUANTUM ONLY#####################################################
@app.get('/qoptimizationml')
async def qoptimizationml():  # token: str = Depends(oauth2_scheme)
    # async with aiohttp.ClientSession(trust_env=True) as session:
    # async with session.get(url=yarl.URL("https://127.0.1.1:8081"), ssl=False) as response:
    # assert response.status == 200

    import time

    from importlib import reload

    reload(time)
    # measure process time
    t0 = time.clock()
    #import sentiment_analytics
    import stock_data
    #import stoch
    #reload(sentiment_analytics)
    #time.sleep(10)
    reload(stock_data)
    #time.sleep(10)
    #reload(stoch)



    # result = stock_data.TradeDate().run()

    time.clock() - t0, "seconds process time"
    await stock_data.MainApp().run()
    from psutil import process_iter
    from signal import SIGTERM  # or SIGKILL

    for proc in process_iter():
        for conns in proc.connections(kind='inet'):
            if conns.laddr.port == 8001:
                proc.send_signal(SIGTERM)  # or SIGKILL
                continue
            else:
                pass
    from subprocess import Popen
    p = Popen("openpage.bat", cwd=r"C:\\Users\User\xxxx")
    # stdout, stderr = p.communicate()

    import webbrowser

    webbrowser.get()
    webopen = webbrowser.open("https://xxxxxx.xxxx.xx/strategy/19")
    return RedirectResponse(url=f"/strategy/19",status_code=303)

##########################################################################
#For classical optimization
##########################################################################
@app.get('/classicaloptimization')
async def classicaloptimization(): # token: str = Depends(oauth2_scheme)
    #async with aiohttp.ClientSession(trust_env=True) as session:
        #async with session.get(url=yarl.URL("https://127.0.1.1:8081"), ssl=False) as response:
            #assert response.status == 200
    import time
    from importlib import reload
    reload(time)

    # measure process time
    t0 = time.clock()
    import classical_stock_data
    reload(classical_stock_data)

    time.clock()-t0, "seconds process time"

    import webbrowser

    webbrowser.get(using=None)
    return (webbrowser.open('https://xxxx.xxxx.xx/strategy/19'))


#############################################################################
#For Machine Learning
#############################################################################

@app.get('/machinelearning')
async def machinelearning(): # token: str = Depends(oauth2_scheme)
    #async with aiohttp.ClientSession(trust_env=True) as session:
        #async with session.get(url=yarl.URL("https://127.0.1.1:8081"), ssl=False) as response:
            #assert response.status == 200
    import time
    from importlib import reload
    reload(time)
    # measure process time
    t0 = time.clock()
    import sentiment_analytics
    reload(sentiment_analytics)


    time.clock()-t0, "seconds process time"
    #await sentiment_analytics
    import webbrowser

    webbrowser.get(using=None)
    return (webbrowser.open('https://xxxxx.xxxx.xx/strategy/19'))



    # The website will open.
    #open_url('https://xxxxx.xxxx.xx/strategy/19')

'''
"Machine Learning Sentiment Analysis on classical computer was successfully done in ", time.clock()-t0, "seconds process time"
'''
#########################################################################################################

##########################FOR UPDATING CHARTS DATABASE###################################################
@app.get('/chartsIndicators')
async def chartsIndicators(): # token: str = Depends(oauth2_scheme)
    #async with aiohttp.ClientSession(trust_env=True) as session:
        #async with session.get(url=yarl.URL("https://127.0.1.1:8081"), ssl=False) as response:
            #assert response.status == 200

    import time

    from importlib import reload


    reload(time)
    # measure process time
    t0 = time.clock()
    import stoch
    reload(stoch)

    #result = stock_data.TradeDate().run()
    #print
    time.clock()-t0 #, "seconds process time"
    import webbrowser

    webbrowser.get(using=None)
    return (webbrowser.open('https://xxxxx.xxxxx.xx/strategy/19'))
    #await stoch
    #from psutil import process_iter
    #from signal import SIGTERM  # or SIGKILL


    #for proc in process_iter():
    #    for conns in proc.connections(kind='inet'):
    #        if conns.laddr.port == 8001:
    #            proc.send_signal(SIGTERM)  # or SIGKILL
    #            continue
    #        else:
    #            pass
    #from subprocess import Popen
    #p = Popen("openpage.bat", cwd=r"C:\\Users\User\xxxx")
    #stdout, stderr = p.communicate()


    #stock_data.CloseApp().run()
    #return templates.TemplateResponse("strategy.html",{"request": request, "stocks": stocks, "strategy": strategy})

    #p = Popen("StartServer.bat", cwd=r"C:\\Users\User\xxxx")
    #return "Chat data was successfully updated in database in ", time.clock()-t0, "seconds process time"

   


#############################################################################
#For Automatic Trading
@app.get('/automatictrading')
async def automatictrading(request: Request): # token: str = Depends(oauth2_scheme)
    #async with aiohttp.ClientSession(trust_env=True) as session:
        #async with session.get(url=yarl.URL("https://127.0.1.1:8081"), ssl=False) as response:
            #assert response.status == 200
    import time
    from importlib import reload
    reload(time)
    # measure process time
    t0 = time.clock()
    import read_signal
    reload(read_signal)


    time.clock()-t0, "seconds process time"

    #await sentiment_analytics
    some_file_path = "mix.mp3"

    def iterfile():
        with open(some_file_path, mode="rb") as file_like:
            yield from file_like

    time.sleep(10)
    #upload audio to github
    from github import Github
    import requests

    username = "your username"
    # token = "your token"
    token = "your token"
    GITHUB_REPO = "winners"
    # createtheGHclientcorrectly
    g = Github(login_or_token=token)
    # create an instance of an AuthenticatedUser, still without actually logging in
    user = g.get_user()
    # print(user)  # will print 'AuthenticatedUser(login=None)'
    # now, invoke the lazy-loading of the user
    login = user.login
    #print(user)  # will print 'AuthenticatedUser(login=<username_of_logged_in_user>)'
    #print(login)  # will print <username_of_logged_in_user>
    repos = user.get_repos()
    print(repos)
    for repo in repos:
        if repo.full_name == "xxxx/winners":
            print(repo)

            all_files = []
            contents = repo.get_contents("")
            print("Content found")
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    contents.extend(repo.get_contents(file_content.path))
                else:
                    file = file_content
                    all_files.append(str(file).replace('ContentFile(path="', '').replace('")', ''))
            print(all_files)

            with open("mix.mp3", 'rb') as file:
                # with open(all_files[2], 'r') as file:
                content = file.read()
                #print(content)

            # Upload to github
            git_prefix = ""
            git_file = git_prefix + "mix.mp3"

            print(git_file)
            if git_file in all_files:
                repo = g.get_repo("xxxx/winners")
                contents = repo.get_contents(git_file, ref="main")

                print(contents.sha)
                print(contents)
                print(contents.name)
                print(repo)
                repo.update_file(path=contents.path, message="committing files", content=content, sha=contents.sha,
                                 branch="main")

                print("repository updated")
                print(git_file + ' UPDATED')

            else:
                repo.create_file(git_file, "committing files", content, "main")
                print(git_file + ' CREATED')


    import webbrowser

    webbrowser.get(using=None)
    return (webbrowser.open('https://xxxx.xxxx.xx/strategy/19'))
    #return StreamingResponse(iterfile(), media_type="audio/mp3")


    #return templates.TemplateResponse("automatictrading.html", {"request": request})
    #return "Automatic trading signal generated successfully in ", time.clock()-t0, "seconds process time"

########################Progress bar#####################################################


"""
    async def async_generator():
        for i in range(3):
            await asyncio.sleep(1)
            yield i*i


    async def main():
        async for i in async_generator():
            print(i)


    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())  # see: https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.shutdown_asyncgens
        loop.close()
"""




"""
function_name=stock_data

def authenticated_sanitizer(token: str = Depends(token_auth_scheme),
                            function_name: str = Path(..., title="stock_data")):
    # check for credentials first
    if not token.credentials == "123456789":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": token.scheme},
        )
    sanitized_fname = html.escape(function_name)
    slugified_filename = slugify(sanitized_fname, to_lower=True, separator="_")
    slugified_fullname = f"{slugified_filename}.py"

    if isfile(slugified_fullname):
        return slugified_fullname
    else:
        raise HTTPException(status_code=404,
                            detail=f"{slugified_fullname} not found")
@app.get(f"/authenticated_sanitized_magic/{function_name}/", dependencies=[Depends(RateLimiter(times=2, seconds=5))])
async def public(function_name: str = Depends(authenticated_sanitizer)):
    result = subprocess.check_output([sys.executable, function_name, "34"])
    return {
        "file_name": f"{function_name}.py",
        "result": result
    }
import sys


policy = asyncio.WindowsSelectorEventLoopPolicy()
asyncio.set_event_loop_policy(policy)
loop = asyncio.get_event_loop()


#session = aiohttp.ClientSession(trust_env=True)
@app.on_event("startup")
async def startup():
    async with aiohttp.ClientSession(trust_env=True,loop=loop) as session:

        async with session.get(url=yarl.URL("http://127.0.1.1:8081"), ssl=False) as response:
            assert response.status==200
            redis = await aioredis.from_url(url="redis://127.0.1.1:8081/stock_data.py")#, encoding="utf-8", decode_responses=False)

            #http= aiohttp.ClientResponse(url=yarl.URL("http://127.0.1.1:8081/stock_data.py"),method="GET", writer=None, continue100= 1, timer= None,request_info=None,traces=None,loop=loop, session=session)

            await FastAPILimiter.init(redis)

"""





