# MLFlaskapp
Machine learning classification model and flask web app



The goal of this project was to design, develop and implement a machine learning algorithm for a fictional company. In addition, the following criteria had to be met:

• one descriptive method and one non-descriptive (predictive or prescriptive) method

•  collected or available datasets

•  decision-support functionality

•  ability to support featurizing, parsing, cleaning, and wrangling datasets

•  methods and algorithms supporting data exploration and preparation

•  data visualization functionalities for data exploration and inspection

•  implementation of interactive queries

•  implementation of machine-learning methods and algorithms

•  functionalities to evaluate the accuracy of the data product

•  industry-appropriate security features

•  tools to monitor and maintain the product

•  a user-friendly, functional dashboard that includes at least three visualization types

The data for this project was collected from the following website:
https://www.kaggle.com/andrewmvd/fetal-health-classification

It was used to create a web application that accepts medical parameters based on the variables on the data set, and returns fetal classification: healthy, suspect, or pathological using Random Forest Classifier. Additionally, I've implemented a dashboard that allows the user to compare variable distribution, view correlations between best features, and explore the accuracy of the chosen algorithm compared to other classification algorithms I've tried. There are also graphs exploring cross-validation scores and best features. The graphs were implemented using Plotly Express, Pandas, Matplotlib and Seaborn python libraries. The machine learning model was implemented using NumPy. The entire project was coded in python and the web application was implemented using Flask, and Bootstrap HTML/CSS library. The dashboard includes histograms, scatter plots, 3d scatterplot, bar graphs, heatmap, and a confusion matrix. 

User authentication was implemented with a login screen upon visiting the link. 

Finally, the web application contains a link to a Jupyter Notebook which has additional graphical representations as well as shows how the data was prepared, how the algorithm was chosen from three different classification algorithms, as well as how the chosen algorithm was hypertuned. Finally, The algorithm was cross validated and the original scores and cross validated scores were  compared. 

To view the web application please follow this link:
http://dkrutye.pythonanywhere.com/

Username : admin, Password: admin
