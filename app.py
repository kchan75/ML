from flask import Flask, render_template, request, session, redirect, url_for, make_response
from stock.services import user_service, news_service
from train import train_news
import socket

# -*- coding: utf-8 -*-
app = Flask(__name__, template_folder='html')

@app.route('/', methods=['GET', 'POST'])
def index():
    return  render_template('index.html')

@app.route('/info')
def info():
    return  render_template('info.html')

@app.route('/chart_details', methods=['GET','POST'])
def detail():
    code = request.form['code'][-6:]
    news_data = news_service.get_data(code)
    res_data = {'image':request.form['image'],
                'codeName':request.form['codeName']}

    return render_template('chart_detail.html', res_data=res_data, news_data=news_data)

@app.route('/run')
def run():
    return  render_template('index.html')

app.secret_key = "kChan"
app.run(host='0.0.0.0', port=8087)