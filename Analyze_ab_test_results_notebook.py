#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results Project
# 
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# This project will analyze the results of an A/B test run by an e-commerce company. The aim of the tests will be to see whether the new webpage developed by the company will encourage more users to convert i.e. to start paying for the company's product.
# 
# Three different ways to assess whether the new webpage makes a difference or not will be shown in this project with a conclusion at the end as to what will be best for the company to do in terms of keeping the old page, implementing the new page or running the experiment for longer.
# 
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, we import the necessary libraries.

# In[38]:


import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
random.seed(42)


# a. Read in the dataset and take a look at the top few rows here:

# In[39]:


df=pd.read_csv("ab_data.csv")
df.head()


# b. Number of rows in dataset df.

# In[40]:


len(df)


# c. The number of unique users in the dataset.

# In[41]:


df.user_id.nunique()


# d. The proportion of users converted.

# In[42]:


df.converted.mean()


# e. The number of times the `new_page` and `treatment` don't match.

# In[43]:


df.query('(group == "treatment" and landing_page != "new_page") or (group != "treatment" and landing_page == "new_page")').count()


# f. Checking if any rows have missing values

# In[44]:


df.isnull().sum() 
#no rows have missing values


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Thus we delete all the rows were there is a mismatch.
# 
# a. Created a new dataset **df2** that does not include the rows that were a mismatch

# In[45]:


df2 =df.drop(df.query('(group == "treatment" and landing_page != "new_page") or (group != "treatment" and landing_page == "new_page")').index)
len(df) - len(df2) 
#the non-matching rows from df were dropped in new dataframe df2


# In[46]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# In[47]:


len(df2) #number of rows in dataset df2


# `3.`

# a. Checking how many unique **user_id**s are in **df2**?

# In[48]:


df2.nunique()
# 290584 unique user_ids


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[49]:


df2[df2.duplicated('user_id')]
#user_id 773192 is repeated in df2


# c. What is the row information for the repeat **user_id**? 

# In[50]:


df2[df2.duplicated('user_id')]


# In[51]:


df2.query('user_id == 773192')
# the two rows that have repeated user_id are row 1899 and row 2893


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[52]:


df2.drop(index=2893,inplace=True) #drop repeated row 2893
df2.nunique()


# `4.`
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[53]:


df2.converted.mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[54]:


df2.query('group =="control"').converted.mean()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[55]:


df2.query('group =="treatment"').converted.mean()


# d. What is the probability that an individual received the new page?

# In[56]:


len(df2.query('landing_page == "new_page"'))/len(df2['landing_page'])
#number of rows that received new page divided by total of new and old page received


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# **Explanation**
# 
# The probability for converting for the control group is higher than for the treatment group. The probability of converting reagrdless of which page was received is also higher than the probability associated with the treatment group. 
# This shows us that there is insufficient evidence for us to conclude that the new treatment page leads to more conversions as the probability of conversion is although only slightly lower than for the control group, it is regardless still lower.
# 
# The probability of an individual receiving the new page is 50% thus also 50% chance of receiving the old page so the selection is fair.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# null hypothesis: 𝑝_𝑜𝑙𝑑 >= 𝑝_𝑛𝑒𝑤
# 
# alternative hypothesis: 𝑝_𝑜𝑙𝑑 < 𝑝_𝑛𝑒𝑤
# 

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[57]:


p_new = 0.1196


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[58]:


p_old = 0.1196


# In[59]:


df2.head() # have a look at the data again


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[60]:


n_new = len(df2.query('group == "treatment"'))
n_new


# d. What is $n_{old}$, the number of individuals in the control group?

# In[61]:


n_old = len(df2.query('group == "control"'))
n_old


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[62]:


new_page_converted = np.random.choice([0,1],n_new, p=(p_new,1-p_new))
new_page_converted


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[63]:


old_page_converted = np.random.choice([0,1],n_old, p=(p_old,1-p_old))
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[64]:


p_new_minus_old = new_page_converted.mean() - old_page_converted.mean()
p_new_minus_old


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[65]:


p_diffs = []
for _ in range(10000):
    new_page_converted = np.random.choice([0, 1], size=n_new, p=[1-p_new, p_new])
    old_page_converted = np.random.choice([0, 1], size=n_old, p=[1-p_old, p_old])
    diff = new_page_converted.mean() - old_page_converted.mean()
    p_diffs.append(diff)
p_diffs = np.array(p_diffs)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[66]:


plt.hist(p_diffs);
# this plot does look like what i expected which was a normal distribution
#looking graph with the centre centered around zero.


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[67]:


# actual differences observed, obs_diff are computed as:
obs_diff = df2.query('group == "treatment"').converted.mean() - df2.query('group == "control"').converted.mean()

(p_diffs > obs_diff).mean()
# 90.74% of p_diffs are greater than actual difference observed


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **Explanation**
# 
# Part j computed the p-value, which is the probability of obtaining the observed results of a test assuming the null is true.
# Large p-values imply that we should not move away from the null hypothesis, thus in this case with such a large p-value of 0.9073, we would fail to reject the null hypothesis and conclude that the treatment new page is not better than the old.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[68]:


convert_old = sum((df2.group == 'control') & (df2.converted == 1)) 
convert_new = sum((df2.group == 'treatment') & (df2.converted == 1)) 
n_old = n_old
n_new = n_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[69]:


z_score, p_value = sm.stats.proportions_ztest(
    [convert_new, convert_old], 
    [n_new, n_old], alternative='larger') 
z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Explanation**
# 
# The p-value here of 0.905 is very close to the p-value obtained in part j and thus we would reject the null hypothesis and conclude that the treatment new page is not better than the old.
# A z test is run to show whether two population means are different and in this case, using one-tailed 5% significance, from the z score table we obtain a test statistic of 0.885.
# Comparing this to the z-score of -1.31, z-score is less than the test statistic and thus we would fail to reject the null hypothesis and conclude that treatment new page is not better than the old page.
# 
# Thus our findings for part m. do agree with the findings in parts j. and k.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# Logistic Regression because the response variable is categorical that can only predict two possible outcomes.

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[70]:


df2.head(2)


# In[71]:


df2['intercept']=1 #creating the intercept
df2['ab_page'] = pd.factorize(df2.group)[0] #creating dummy variables

df2.head() #checking if the two columns were added correctly to df2


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[72]:


logit_model = sm.Logit(df2['converted'],df2[['intercept','ab_page']])
results=logit_model.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[76]:


results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# **Explanation**
# 
# The hypothesis for the regression are:
# null hypothesis: diffrence in conversion rates = 0
# alternative hypothesis: diffrence in conversion rates != 0
# 
# This differs from Part II, because in part II, one tailed test was used and for the regression a two-tailed test is used.
# 
# The p-value for the regression is 0.19 which differs from the p-value in part II of 0.90 and this is due to the difference in one-tailed vs. two-tailed test used as mentioned above.

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Explanation**
# 
# Other things that may influence if an individual converts or not is how much time they spend on the page and on which days of the week. How long a client has used the company's product previously or if at all before they had to convert to paying for it.
# 
# Some disadvantages to adding additional terms to a regression model may be issues associated with multicollinearity.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[77]:


df_countries = pd.read_csv('countries.csv')
df_countries.head(2) #look at data


# In[78]:


df_countries.nunique() # 3 different countries


# In[79]:


df3 = df2.set_index('user_id').join(df_countries.set_index('user_id'))
#used user_id to join the two datasets
df3.head(2) #checking if join was successful and correct


# In[80]:


df3.country.value_counts()


# In[81]:


df3[['CA','UK','US']] = pd.get_dummies(df3['country']) #create dummy variables
df3.head() #check dummy variables were created correctly


# In[82]:


# Use CA as baseline country
logit_model = sm.Logit(df3['converted'],df3[['intercept','ab_page','US','UK']])
results=logit_model.fit()
results.summary()


# **Explanation**
# 
# Looking at the p-values from this regression that are higher than 5%, all the country variables are not statistically significant thus there is no effect based on the country a user lives in.
# ab_page variable also has a p-value higher than 5% as well and thus we fail to reject the null hypothesis and conclude that there is no difference in conversion rates between the two different pages, old and new.

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[83]:


df3['ab_page_US'] = df3['ab_page']*df3['US'] #US and conversion interaction term
df3['ab_page_UK'] = df3['ab_page']*df3['UK'] #UK and conversion interaction term


# In[84]:


logit_model2 = sm.Logit(df3['converted'],df3[['intercept','ab_page','US','UK','ab_page_US','ab_page_UK']])
results=logit_model2.fit()
results.summary()


# **Explanation**
# 
# In this regression output the interaction terms between country and page have been included and yet again all the p-values being higher than 5% show that the variables are not statistically significant and thus there is no effect on a page choice that depends on the country the user is from.
# Country may not have an effect as this is an e-commerce website and thus customers have access to it from anywhere in the world.

# In[86]:


df2.timestamp.min(), df2.timestamp.max() #checking time length of experiment. it is roughly 3 weeks


# **CONCLUSION**
# 
# Looking at all three sections, probability, A/B test and regression. It can be concluded that all three sections agree and show that the treatment new page would not be better at increasing conversion rates than the old page and thus the company should keep the old page.
# However it would seem to be beneficial to run the experiment for a bit longer as this experiment was only run for less than a month and that may be too short a time to really assess the impact of the new page especially on new and existing customers.

# In[87]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

