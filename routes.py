from audioop import cross
from __init__ import app
from flask import render_template, url_for, redirect, request, jsonify
import pandas as pd
import json
import plotly
from plotly import graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np



@app.route('/', methods=['GET', 'POST'] )
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('index'))
    return render_template('login.html', error=error)

@app.route ("/analyze")
def analyze():
    return render_template ('analyze.html')

from machine import clf1
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = clf1.predict(final_features)

    output = round(prediction[0], 21)

    return render_template('analyze.html', prediction_text='The baby is {}'.format(output))


@app.route("/index")
def index():
    #graph one

    df = pd.read_csv("fetal_health.csv")
    histogram = px.histogram(df, x="baseline_value", title="FHR baseline (beats per minute)", width=1000, height=400,
    labels = {
                "baseline_value" : "BPM"}         
                )
    histogram.update_layout(paper_bgcolor="#E9C46A")
    hist1JSON = json.dumps(histogram, cls = plotly.utils.PlotlyJSONEncoder)

    #histogram 2

    histogram2 = px.histogram(df, x="accelerations", title="Number of accelerations per second", width=1000, height=400,
    labels = {
                "accelerations" : "accelerations per second"})
    histogram2.update_layout(paper_bgcolor = "#E9C46A")
             
    hist2JSON = json.dumps(histogram2, cls = plotly.utils.PlotlyJSONEncoder)

    #histogram 3

    histogram3 = px.histogram(df, x="fetal_movement", title="Fetal Movement", width=1000, height=400,
    labels = {
                "fetal_movement" : "Number of fetal movements per second"})
    histogram3.update_layout(paper_bgcolor = "#E9C46A")
             
    hist3JSON = json.dumps(histogram3, cls = plotly.utils.PlotlyJSONEncoder)

    #histogram 4

    histogram4 = px.histogram(df, x="uterine_contractions", title="Number of uterine contractions per second", width=1000, height=400,
    labels = {
                "uterine_contractions" : "uterine contractions per second"})
    histogram4.update_layout(paper_bgcolor = "#E9C46A")
             
    hist4JSON = json.dumps(histogram4, cls = plotly.utils.PlotlyJSONEncoder)

    #histogram 5

    histogram5 = px.histogram(df, x="light_decelerations", title="Number of light decelerations per second", width=1000, height=400,
    labels = {
                "light_decelerations" : "light decelerations per second"})
    histogram5.update_layout(paper_bgcolor = "#E9C46A")
             
    hist5JSON = json.dumps(histogram5, cls = plotly.utils.PlotlyJSONEncoder)

    #histogram 6

    histogram6 = px.histogram(df, x="severe_decelerations", title="Number of severe decelerations per second", width=1000, height=400,
    labels = {
                "severe_decelerations" : "severe decelerations per second"})
    histogram6.update_layout(paper_bgcolor = "#E9C46A")
             
    hist6JSON = json.dumps(histogram6, cls = plotly.utils.PlotlyJSONEncoder)

    #histogram 7

    histogram7 = px.histogram(df, x="prolongued_decelerations", title=" Number of prolonged decelerations per second", width=1000, height=400,
    labels = {
                "prolongued_decelerations" : "prolonged decelerations per second"})
    histogram7.update_layout(paper_bgcolor = "#E9C46A")
             
    hist7JSON = json.dumps(histogram7, cls = plotly.utils.PlotlyJSONEncoder)

    #histogram 8

    histogram8 = px.histogram(df, x="abnormal_short_term_variability", title="Time with abnormal short term variability", width=1000, height=400,
    labels = {
                "abnormal_short_term_variability" : "Percentage of time with abnormal short term variability"})
    histogram8.update_layout(paper_bgcolor = "#E9C46A")
             
    hist8JSON = json.dumps(histogram8, cls = plotly.utils.PlotlyJSONEncoder)

    #histogram 9

    histogram9 = px.histogram(df, x="mean_value_of_short_term_variability", title="Mean value of short term variability", width=1000, height=400,
    labels = {
                "mean_value_of_short_term_variability" : "Mean value of short term variability"})
    histogram9.update_layout(paper_bgcolor = "#E9C46A")
             
    hist9JSON = json.dumps(histogram9, cls = plotly.utils.PlotlyJSONEncoder)

    #histogram 10

    histogram10 = px.histogram(df, x="percentage_of_time_with_abnormal_long_term_variability", title="Time with abnormal long term variability", width=1000, height=400,
    labels = {
                "percentage_of_time_with_abnormal_long_term_variability" : "Percentage of time with abnormal long term variability"})
    histogram10.update_layout(paper_bgcolor = "#E9C46A")
             
    hist10JSON = json.dumps(histogram10, cls = plotly.utils.PlotlyJSONEncoder)

    # histogram 11

    histogram11 = px.histogram(df, x="mean_value_of_long_term_variability", title="Mean value of long term variability", width=1000, height=400,
    labels = {
                "mean_value_of_long_term_variability" : "Mean value of long term variability"})
    histogram11.update_layout(paper_bgcolor = "#E9C46A")
             
    hist11JSON = json.dumps(histogram11, cls = plotly.utils.PlotlyJSONEncoder)

    #scatter1
    scatter = px.scatter_3d(df, x="abnormal_short_term_variability", color = "fetal_health",  z="percentage_of_time_with_abnormal_long_term_variability", y = "mean_value_of_short_term_variability", labels = {
                "abnormal_short_term_variability" : "abnormal short", "percentage_of_time_with_abnormal_long_term_variability": "abnormal long", "mean_value_of_short_term_variability": "short term mean"}, title = "Comparing Fetal Health, Abnormal Short Term Variability, and Mean Value of STV", width=1000, height=400 )
    scatter.update_layout(paper_bgcolor = '#E9C46A')
    scatterJSON = json.dumps(scatter, cls = plotly.utils.PlotlyJSONEncoder)

    #scatter 2

    scatter2 = px.scatter(df, x = "histogram_median", y = "accelerations", color = "fetal_health", title = "Histogram Median and Accelerations", labels = {
                "histogram_median" : "Histogram median", "fetal_health": "fetal health"}, width=1000, height=400)
    scatter2.update_layout(paper_bgcolor = '#E9C46A')
    scatter2JSON = json.dumps(scatter2, cls = plotly.utils.PlotlyJSONEncoder)

    #scatter 3
    
    scatter3 = px.scatter(df,  y = "fetal_movement", x = "uterine_contractions", labels = {"fetal_movement" : "fetal movement", "uterine_contractions":"uterine contractions"}, color = "fetal_health", title = "Prolonged Decelerations and BPM", width=1000, height=400)
    scatter3.update_layout(paper_bgcolor = '#E9C46A')
    scatter3.update_traces(marker=dict(size=14, opacity = .5,
                              line=dict(width=0,
                                        )),
                  selector=dict(mode='markers'))

    scatter3JSON = json.dumps(scatter3, cls = plotly.utils.PlotlyJSONEncoder)

    #heat map   
    
    corrmatrix = df.corr()
    fix, ax = plt.subplots(figsize = (15,15))
    x_axis_labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22] # labels for x-axis
    y_axis_labels = [22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]  # labels for y-axis
    sns.set_style('dark')

    zx = sns.heatmap(corrmatrix, annot = True, xticklabels=x_axis_labels, yticklabels=y_axis_labels, linewidths = 0.5, fmt = ".2f").set(title = 'Correlation between Medical Parameters')
    bottom, top = ax.get_ylim()
    plt.savefig('static/corrmatrix.png', bbox_inches='tight', facecolor = '#E9C46A')


    #bar graph model scores
    from machine import model_scores as model_scores


    model_compare = pd.DataFrame(model_scores, index = ["algorithm"])

    modelbar = px.bar(model_compare, x = ['Logistic Regression', 'KNN', 'Random Forest'], barmode='group', title = "Initial Algorithm Accuracy",width=1000, height=400 )
    modelbar.update_layout(paper_bgcolor = '#E9C46A')

    modelbarJSON = json.dumps(modelbar, cls = plotly.utils.PlotlyJSONEncoder)

    # confusion matrix for the hypertuned model
    from machine import y_prediction, y_test
    def confusion(y_test, y_prediction):
        """
        confusion matrix using seaborn's heatmap
        """
    fig, ax = plt.subplots(figsize = (5,5))
    ax = sns.heatmap(confusion_matrix(y_test, y_prediction),
                    annot = True,
                    cbar = False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Number of Correct and Incorrect Predictions")
    
    confusion(y_test, y_prediction)
    plt.savefig('static/confusion.png', bbox_inches='tight', facecolor = '#E9C46A')


    ## SCORE COMPARISONS##
    # import scores
    from machine import accuracy_scr, preciss, recc, f1sc, cv_acc, cv_precision, cv_recall, cv_f1

    #graph for original score

    original_score = pd.DataFrame({"Accuracy": accuracy_scr, "Cross Accuracy":cv_acc, "Precision": preciss,  "Cross Precision":cv_precision,  "Recall": recc, "Cross Recall": cv_recall , "F1 Score":f1sc, "Cross F1": cv_f1, }, index = [0])

    originalbar = px.bar(original_score, title = "Test Scores vs Cross Validation", labels = {'index': 'type of score', 'value' : "accuracy percentage" }, barmode='group', width=1000, height=400)
    originalbar.update_layout(paper_bgcolor = '#E9C46A')
    originalbarJSON = json.dumps(originalbar, cls = plotly.utils.PlotlyJSONEncoder)

    #feature importances
    from machine import feat_importances, features, X
    imp_pd = pd.DataFrame(features, index = X.columns)
    imp_pd.rename(
   index={"histogram_variance" : "Histogram variance", "baseline_value" : "baseline value", "fetal_movement" : "fetal movement","uterine_contractions" : "contractions", "light_decelerations" : "light decelerations", "severe_decelerations" : "severe decelerations","prolongued_decelerations" : "prolongued decelerations", "abnormal_short_term_variability" : "abnormal variability","mean_value_of_short_term_variability" : "short term mean","percentage_of_time_with_abnormal_long_term_variability" : "time with abnormal LT","mean_value_of_long_term_variability" : "long term mean","histogram_width" : "histogram width","histogram_min" : "histogram min","histogram_max" : "histogram max","histogram_number_of_peaks" : "histogram peaks","histogram_number_of_zeroes" : "histogram zeroes",
    "histogram_mode" : "histogram mode",
    "histogram_mean" : "histogram mean",
    "histogram_median" :"histogram mediam",
    "histogram_tendency" :"histogram tendency" }
          ,inplace=True)
    importancebar = px.bar(imp_pd, title = "Importance of Medical Variables on Fetal Health", barmode ='group', labels = {'index': 'Medical Parameter', 'value' : "Importance"}, width=1000, height=400 )
    importancebar.update_layout(paper_bgcolor = '#E9C46A', showlegend = False)
    importancebar.update_xaxes(type = 'category')
    importancebarJSON = json.dumps(importancebar, cls = plotly.utils.PlotlyJSONEncoder)
    
    return render_template("index.html", title = "Home", hist1JSON = hist1JSON, hist2JSON = hist2JSON, hist3JSON = hist3JSON, hist4JSON = hist4JSON, hist5JSON = hist5JSON,
                            hist6JSON = hist6JSON, hist7JSON = hist7JSON, hist8JSON = hist8JSON, hist9JSON = hist9JSON, hist10JSON = hist10JSON, hist11JSON = hist11JSON, scatterJSON = scatterJSON, scatter2JSON = scatter2JSON, scatter3JSON = scatter3JSON, modelbarJSON = modelbarJSON, originalbarJSON=originalbarJSON, importancebarJSON = importancebarJSON)

@app.route('/jupyter')
def jupyter():
    return redirect('https://mybinder.org/v2/gh/DaryaKrutyeva/classificationjupyter/HEAD')
             
    


    