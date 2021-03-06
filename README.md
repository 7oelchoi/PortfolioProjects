<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

![](Banner.svg)


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-repository">About the Repository</a>
    </li>
    <li>
      <a href="#CCF">Credit Card Fraud/a>
    </li>
    <li>
      <a href="#COVID-Project">COVID Project</a>
    </li>
    <li>
      <a href="#Cyclistic-Project">Cyclistic Project</a>
    </li>
    <li>
      <a href="#Housing-Project">Housing Project</a>
    </li>
    <li>
      <a href="#WebScraping-Project">Webscraping Project</a>
    </li>
  </ol>
</details>



<!-- ABOUT THE REPOSITORY -->
## About the repository

This repository is a collection of projects I have conducted in my spare time. This portfolio aims to give a brief overview on my capabilities as a Data analytics specialist.
I have tried to use a multitude of different languages and tools in different projects.

Some competencies displayed here are:
* Creating queries to get specific data (Temp tables, CTE, Views)
* Using R for data import, cleaning, exploration, and analysis
* R Markdown for documentation on analysis process
* Tableau for creating interactive dashboards

In the following a short introduction to each project is shown, with regards to the programming languages and tools used.
For detailed implementation, feel free to look inside the folders!

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CCF -->
## Credit Card Transactions Fraud

 [Kernel](https://github.com/7oelchoi/PortfolioProjects/blob/main/Creditcard_fraud/Credit%20Card%20Fraud.ipynb) | Python | Machine Learning 
 --- | --- | ---

This data set is derived from kaggle and offers an insight to fraudulent and valid credit card transactions. It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

The Kernel contains an exploration and balancing of the dataset, and the training and evaluation of models using different classification methods.

<!-- COVID PROJECT -->
## COVID Project

Data Analysis | [SQL](https://github.com/7oelchoi/PortfolioProjects/blob/main/COVID/Queries/COVID_Portfolio.sql) | [Tableau](https://github.com/7oelchoi/PortfolioProjects/blob/main/COVID/Visualization/Covid_Dashboard.png)
--- | --- | --- 

This project uses a data set on worldwide COVID-19 numbers. Especially, infections, deaths, and vaccinations are listed. In the project, the data is inspected, by comparing infection rates in regions, death rates and the correlation to vaccinations.
After exploring the data, a dashboard is created using Tableau to provide a general overview and a tool for the user to interactively explore the data set.
The data is derived from https://ourworldindata.org/covid-deaths and is pre cleaned and ready to use.

For the data exploration SQL has been used with Microsoft SQL Server Management Studio. Tableau was used to create the dashboard. The data is represented by csv files.

There is not a separate documentation of the steps taken, however each query is commented with thourough explanation.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CYCLISTIC PROJECT -->
## Cyclistic Project

 [Full Project](https://github.com/7oelchoi/PortfolioProjects/blob/main/Cyclistic/Cyclistic_documentation.pdf) | [R](https://github.com/7oelchoi/PortfolioProjects/blob/main/Cyclistic/Cyclistic.R)  
 --- | ---

The cyclistic project is a capstone project suggested in the Google Data Analytics Professional Certificate course. 
In this project I take data from https://divvy-tripdata.s3.amazonaws.com/index.html, and perform 5 data analytics steps:

1. Ask
2. Prepare
3. Process
4. Analyze
5. Act

A thorough documentations on each step is provided in https://github.com/7oelchoi/PortfolioProjects/blob/main/Cyclistic/Cyclistic_documentation.html.

Here the language R is used with RStudio and Markdown for the documentation. Visualizations are created by using the ggplot2 package. Tidyverse and multiple other packages are used too.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- HOUSING PROJECT -->
## Housing Project

[Data Cleaning](https://github.com/7oelchoi/PortfolioProjects/blob/main/Housing/Documentation.pdf)  | [SQL](https://github.com/7oelchoi/PortfolioProjects/blob/main/Housing/DataCleaningQueries.sql) 
--- | ---

This project inspects a data set on housing sales. Here, I used SQL to clean the data using various methods. 

I used SQL and the Microsoft SQL Server Management Studio to perform the cleaning steps.
Here are some of the cleaning steps performed:

* Removing unnecessary information from columns
* Splitting columns into multiple different ones
* Populateing NULL values
* Removing inconsistencies
* Removing duplicate entries

A more detailed description on the data cleaning process is shown in https://github.com/7oelchoi/PortfolioProjects/blob/main/Housing/Documentation.pdf. The documentation was done using LaTeX.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- WEBSCRAPING PROJECT -->
## Webscraping Project

Webscraping | [Data Cleaning](https://github.com/7oelchoi/PortfolioProjects/blob/main/WebScraping/Amazon_Webscraper.ipynb)  | Beautifulsoup
--- | --- | ---

Here, I use the library beautiful soup to extract data from a website and use it to create a dataframe. 
It updates periodically and has the functionality of sending an email when the price of an iPad on amazon reaches a certain price.

<p align="right">(<a href="#top">back to top</a>)</p>
