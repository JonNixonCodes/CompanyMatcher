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
        assert(type(company)==str)
        return self.vectoriser.transform([company]).toarray()
    
    def transform_corpus(self, corpus):
        assert(type(corpus)==list)
        return self.vectoriser.transform(corpus).toarray()

    def find_nearest_match(self, company_v, corpus_v, n=1):
        assert(corpus_v.shape[1]==company_v.shape[1])
        dist_v = np.linalg.norm(corpus_v-company_v, axis=1)
        match_idx_l = np.argsort(dist_v)
        match_l = [(idx,dist_v[idx]) for idx in match_idx_l[:n]]
        return match_l

    # No longer used    
    def similarity(self, c1, c2):
        v1 = self.transform(c1)
        v2 = self.transform(c2)
        return np.linalg.norm(v1-v2)

