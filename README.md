# Identifying Fraud from Enron Emails

**Ioannis K Breier**

[Corporate corpus: Volumes of e-mails that were sent and received in Enron’s headquarters in Houston, seen here in 2002, are still parsed and dissected by computer scientists and other researchers.](https://cdn.technologyreview.com/i/images/enronx299.jpg?sw=480)

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.   

This data has been combined with a hand-generated list of persons of interest (POI) in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.  
The dataset, before any transformations, contained __146 records__ consisting of __14 financial features__ (all units are in US dollars), __6 email features__ (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string), and __1 labeled feature (POI)__. 

The aim of this project is to create a model that, using the optimal combination of the available features, can identify  whether a person is a POI or not.    
Since the dataset contains financial and email information that is common among most corporations it could potentially be used to help identify person of interests in similar situations in other companies.

**[Project Report](https://jkarakas.github.io/Identify-Fraud-from-Enron-Email/report.html)**

