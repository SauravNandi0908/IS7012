#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("laliga21-22.csv")
df.head(19)


# In[2]:


df.shape


# In[4]:


df.info()


# In[5]:


df.sort_values("Pts", ascending=False)[["Squad", "Pts"]]


# In[6]:


plt.figure()
plt.scatter(df["GD"], df["Pts"])
plt.xlabel("Goal Difference")
plt.ylabel("Points")
plt.title("Goal Difference vs Points (La Liga 2021–22)")
plt.show()


# In[ ]:


#Strong positive correlation

#Higher goal difference → more points

#One of the best predictors of league success



# In[7]:


plt.figure()
plt.scatter(df["GF"], df["GA"])
plt.xlabel("Goals For")
plt.ylabel("Goals Against")
plt.title("Goals For vs Goals Against")
plt.show()


# In[ ]:


#Top teams score more and concede less

#Defensive stability clearly separates elite teams


# In[8]:


df["Goal_Diff_xG"] = df["GF"] - df["xG"]
df[["Squad", "GF", "xG", "Goal_Diff_xG"]].sort_values("Goal_Diff_xG", ascending=False)


# In[9]:


df[["Squad", "pass completion", "prog dist", "Pts"]].sort_values(
    "pass completion", ascending=False
)


# In[ ]:


# Higher pass completion & progressive distance correlate with:

# Better control

# Higher league positions


# In[10]:


df[["Squad", "fouls", "fouls drawn", "aerial per"]].sort_values("fouls", ascending=False)


# In[11]:


corr = df[["Pts", "GD", "GF", "GA", "xG", "pass completion"]].corr()
corr


# In[ ]:


# Pts & GD → strongest correlation

# Pts & GF → strong

# Pts & GA → negative correlation


# In[12]:


df[["Squad", "fouls", "fouls drawn", "aerial per"]].sort_values("fouls", ascending=False)


# In[ ]:


# Aggressive teams commit more fouls

# High aerial win % often seen in mid-table defensive teams


# In[13]:


df.sort_values("GD", ascending=False)[["Squad", "GD", "Pts"]]


# In[ ]:


# Real Madrid have the highest goal difference, ie it scored the most against all opponents and it conceeded the least amongst the 38  matches they played throughout the season. That is why they earned the highest number of points and won LaLiga that season


# In[14]:


df.sort_values("GF", ascending=False)[["Squad", "GF"]]


# In[ ]:


# Real Madrid  scored the most goals against la liga teams


# In[15]:


df.sort_values("GA")[["Squad", "GA"]]


# In[ ]:


# Real madrid conceeded the least when compared to mid table teams and it scored the most goals in la liga across different opponents and thus, it won la liga 2021/2022 season


# In[16]:


df["Goal_Diff_xG"] = df["GF"] - df["xG"]
df.sort_values("Goal_Diff_xG", ascending=False)[["Squad", "Goal_Diff_xG"]]


# In[17]:


df.sort_values("pass completion", ascending=False)[["Squad", "pass completion"]]


# In[ ]:


# Real madrid performed well in 21/22 season and has completed most passes, indicating high team chemistry compared to other La Liga teams


# In[18]:


df.sort_values("prog dist", ascending=False)[["Squad", "prog dist"]]


# In[ ]:


# Real Madrid had the most completed passes and had the 2nd most progressive carries towardws opponent's goal, with the leaders in this statistic being Barcelona


# In[19]:


df[["Squad", "fouls", "fouls drawn"]].sort_values("fouls")


# In[ ]:


# As you can see from the Chart above, Real Madrid committed the least number of fouls , thus avoiding major suspension of main players from its first team, reducing injuries or risk of injuries and hence
# keeping the squad healthy to compete at later stages of La Liga

#Real Madrid won La Liga primarily due to their superior goal difference, which reflected an elite balance of attacking output and defensive solidity. They combined one of the league’s most effective attacks with a disciplined defense, outperformed expected goals through clinical finishing, and controlled matches using high pass completion and progressive play. This reduced match-to-match volatility and allowed them to consistently accumulate points across the season.

