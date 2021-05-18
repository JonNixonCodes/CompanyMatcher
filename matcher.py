#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Company matcher

Created on Tue May 18 23:00:28 2021

@author: JonNixonCodes
"""

# %% Import libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# %% Matcher
class Matcher:
    
    def __init__(self, corpus):
        self.vectoriser = TfidfVectorizer()
        self.vectoriser.fit(corpus)
    
    def fit(self, corpus):
        self.vectoriser.fit(corpus)
        
    def transform(self, company):
        return self.vectoriser.transform([company]).toarray()
    
    def similarity(self, c1, c2):
        v1 = self.transform(c1)
        v2 = self.transform(c2)
        return np.linalg.norm(v1-v2)
