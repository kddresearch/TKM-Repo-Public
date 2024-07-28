
from flask import Flask, render_template
import pandas as pd
import json

from flask_pymongo import PyMongo
import re


from flask import Flask
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo

#Natural Language Processing (NLP)
import spacy
import gensim
from spacy.tokenizer import Tokenizer
from gensim.models.ldamulticore import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS as SW
# from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.model_selection import GridSearchCV


import time

import sys

# https://towardsdatascience.com/web-visualization-with-plotly-and-flask-3660abf9c946

# ======== Custom modules ========
import lda_functions as LDAFunctions
import hdp_functions as HDPFunctions

# https://stackoverflow.com/questions/25623669/python-get-console-output
class Logger():
    stdout = sys.stdout
    messages = []

    def start(self): 
        sys.stdout = self

    def stop(self): 
        sys.stdout = self.stdout

    def write(self, text): 
        self.messages.append(text)

    def clear(self):
        self.messages = []

# ======== Main Program ========
app = Flask(__name__, static_folder='static')

# =================================
# MongoDB connection information
# =================================
app.config['MONGO_DBNAME'] = 'knowledge-map'
app.config['MONGO_URI'] = 'mongodb://kmAdmin:kmadmin@129.130.10.108:7017/knowledge-map'

mongo = PyMongo(app)
db = mongo.db
etdr = db.etdr_fixed

# =================================
# Render the home page of the flask app. Pulls all departments and advisors from the MongoDB database.
# =================================
@app.route('/')
def index():
    # Departments
    departments = db.etdr_fixed.distinct('dc_description_department')
    departments.sort()
    # Advisors
    advisors = db.etdr_fixed.distinct('dc_description_advisor')
    advisors.sort()
    return render_template('pages/index.html', departments=departments, advisors=advisors)


# KDD Theses Dataframe processing
# Make visualization
@app.route('/visualize', methods=['GET'])
def visualize():
    print("app.py: running visualize")
    start_time = time.time()

    userInput = request.args.get('num_topics', 10)
    num_topics = int(userInput)

    df = pd.DataFrame(list(db.theses.find()))

    graphJSON, topics = LDAFunctions.LDACreatePlotly(df, num_topics, 'text_processed_optimized')
    num_seconds = time.time() - start_time
    num_seconds = str(round(num_seconds, 2))

    return render_template("pages/visualize.html", topics=topics, graphJSON=graphJSON, num_seconds=num_seconds)

@app.route('/visualize_lda', methods=['GET'])
def visualize_lda():
    print("From app.py: /visualize_lda")
    start_time = time.time()

    # get user input for variables, otherwise set default value
    userInput = request.args.get('num_topics', 50)
    num_topics = int(userInput)

    df = pd.DataFrame(list(db.etdr_fixed.find()))

    #graphJSON, topics = create_plotly(df, num_topics, 'abstract_preprocessed')
    model, corpus = LDAFunctions.LDACreateModel(df, num_topics, 'abstract_preprocessed')
    graphJSON, topics = LDAFunctions.LDACreatePlotly(df, model, corpus, num_topics,'abstract_preprocessed' )

    num_seconds = time.time()-start_time
    num_seconds = str(round(num_seconds, 2))
    return render_template("pages/visualize_lda.html", topics=topics, graphJSON=graphJSON, num_seconds=num_seconds)

@app.route('/visualize_hdp', methods=['GET'])
def visualize_hdp():
    print("from app.py: /visualize_hdp")
    start_time = time.time()

    # get parameters from the input forms, otherwise set default values
    min_cf = request.args.get('min_cf', 0)
    min_df = request.args.get('min_df', 0)
    rm_top = request.args.get('rm_top', 0)
    selectedTermWeight = request.args.get('term_weight', 'ONE')
    burn_in = request.args.get('burn_in', 0)
    alpha = request.args.get('alpha', 0.1)
    eta = request.args.get('eta', 0.1)
    gamma = request.args.get('gamma', 0.1)

    df = pd.DataFrame(list(db.etdr_fixed.find()))

    model, topic_labels, top_labels_array = HDPFunctions.HDPCreateModel(df, 
                                                      "abstract_preprocessed",
                                                      termweight=selectedTermWeight,
                                                      min_cf=int(min_cf),
                                                      min_df=int(min_df),
                                                      rm_top=int(rm_top),
                                                      alpha=float(alpha),
                                                      eta=float(eta),
                                                      gamma=float(gamma),
                                                      burn_in=int(burn_in)
                                                       )
    graphJSON, top_labels = HDPFunctions.HDPCreatePlotly(df, model, topic_labels, 'abstract_preprocessed')

    num_seconds = time.time() - start_time
    num_seconds = str(round(num_seconds, 2))

    log = Logger()
    log.clear()
    log.start()
    print(model.summary())
    log.stop()

    allText = log.messages
    model_summary = ""
    for m in allText:
        model_summary = model_summary + m

    return render_template("pages/doc_table.html", #documents=output,
                                 num_seconds=num_seconds,
                                graphJSON=graphJSON, 
                                topics=top_labels, 
                                model_summary=model_summary)



@app.route('/doc_by_department', methods=['GET'])
def doc_by_department():
    start_time = time.time()

    # Get parameters from input form
    selectedDepartment = request.args.get('department_name')
    min_cf = request.args.get('min_cf', 0)
    min_df = request.args.get('min_df', 0)
    rm_top = request.args.get('rm_top', 0)
    selectedTermWeight = request.args.get('term_weight', 'ONE')
    burn_in = request.args.get('burn_in', 0)
    alpha = request.args.get('alpha', 0.1)
    eta = request.args.get('eta', 0.1)
    gamma = request.args.get('gamma', 0.1)

    # Get documents by department and put into HTML table to display
    output = []
    for d in db.etdr_fixed.find({ "dc_description_department": selectedDepartment }):
        output.append({
            'title': d['dc_title'],
            'author': d['dc_contributor_author'],
            'date_published': d['dc_date_published'],
            'type': d['dc_type'],
            'degree': d['dc_description_degree'],
            'department': d['dc_description_department'],
            'advisor': d['dc_description_advisor']
        })

    # Visualize using HDP...
    df = pd.DataFrame(list(db.etdr_fixed.find({ "dc_description_department": selectedDepartment })))
    print(df.head())
    model, topic_labels, top_labels_array = HDPFunctions.HDPCreateModel(df, 
                                                      "abstract_preprocessed",
                                                      termweight=selectedTermWeight,
                                                      min_cf=int(min_cf),
                                                      min_df=int(min_df),
                                                      rm_top=int(rm_top),
                                                      alpha=float(alpha),
                                                      eta=float(eta),
                                                      gamma=float(gamma),
                                                      burn_in=int(burn_in)
                                                       )
    graphJSON, top_labels = HDPFunctions.HDPCreatePlotly(df, model, topic_labels, 'abstract_preprocessed')


    num_seconds = time.time() - start_time
    num_seconds = str(round(num_seconds, 2))

    log = Logger()
    log.clear()
    log.start()
    print(model.summary())
    log.stop()

    allText = log.messages
    model_summary = ""
    for m in allText:
        model_summary = model_summary + m

    return render_template("pages/doc_table.html", documents=output, num_seconds=num_seconds,
                                graphJSON=graphJSON, topics=top_labels, model_summary=model_summary,
                                )

@app.route('/doc_by_advisor', methods=['GET'])
def doc_by_advisor():
    start_time = time.time()
    selectedAdvisor = request.args.get('advisor_name')
    print(selectedAdvisor)

    output = []
    for d in db.etdr_fixed.find({ "dc_description_advisor": selectedAdvisor }):
        output.append({
            'title': d['dc_title'],
            'author': d['dc_contributor_author'],
            'date_published': d['dc_date_published'],
            'type': d['dc_type'],
            'degree': d['dc_description_degree'],
            'department': d['dc_description_department'],
            'advisor': d['dc_description_advisor']
        })
    num_seconds = time.time() - start_time
    return render_template("pages/doc_table.html", documents=output, num_seconds=num_seconds)

# =================================
# LEGACY ROUTES
# =================================
@app.route('/get_kdd_theses', methods=['GET'])
def get_kdd_theses():
    print("app.py: running get_kdd_theses")
    start_time = time.time()
    output = []
    for d in db.etdr_fixed.find().limit(20):
        output.append({
            'title': d['dc_title'],
            'author': d['dc_contributor_author'],
            'date_published': d['dc_date_published'],
            'type': d['dc_type'],
            'degree': d['dc_description_degree'],
            'department': d['dc_description_department'],
            'advisor': d['dc_description_advisor']
        })
    num_seconds = time.time() - start_time
    num_seconds = str(round(num_seconds, 2))
    return render_template("pages/doc_table.html", documents=output, num_seconds=num_seconds)

@app.route('/get_all_theses', methods=['GET'])
def get_all_theses():
    start_time = time.time()
   
    output = []
    for d in db.etdr_fixed.find():
        output.append({
            'title': d['dc_title'],
            'author': d['dc_contributor_author'],
            'date_published': d['dc_date_published'],
            'type': d['dc_type'],
            'degree': d['dc_description_degree'],
            'department': d['dc_description_department'],
            'advisor': d['dc_description_advisor']
        })
    num_seconds = time.time() - start_time
    num_seconds = str(round(num_seconds, 2))
    return render_template("pages/doc_table.html", documents=output, num_seconds=num_seconds)
# ==============================================================

if __name__ == '__main__':
    app.run(debug=True)

