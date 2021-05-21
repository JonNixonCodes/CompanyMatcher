#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Company matcher

Created on Tue May 18 23:00:28 2021

@author: JonNixonCodes
"""

# %% Import libraries
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# %% Matcher
class Matcher:
    """An object for performing company name matching."""
    
    def __init__(self, company_names):
        """Initialise Matcher"""
        self.vectoriser = TfidfVectorizer()
        self.vectoriser.fit(company_names)

    def _convert_vec_to_mat(self, vec, mat):
        """Convert 1xN vector into MxN matrix."""
        data = np.tile(vec.data,16419)
        indices = np.tile(vec.indices,16419) 
        indptr = np.array([i*vec.indptr[1] for i in range(16420)])
        vec_mat = sp.sparse.csr_matrix((data,indices,indptr), shape=mat.shape)
        return vec_mat        
        
    def fit(self, company_names):
        """Fit matcher to a list of company names."""
        return self.vectoriser.fit(company_names)
        
    def transform_one(self, company_name):
        """Return sparse vector representation of company name."""
        return self.vectoriser.transform([company_name])
    
    def transform_many(self, company_names):
        """Return sparse matrix from a list of company names."""
        return self.vectoriser.transform(company_names)

    def find_nearest_match(self, vec, mat, threshold=0.9, n=1):
        """Return N nearest matches between sparse vector and sparse matrix."""
        vec_mat = self._convert_vec_to_mat(vec,mat) # Convert 1xN vector into MxN matrix
        dist = sp.sparse.linalg.norm(mat-vec_mat,axis=1)
        match_idx = np.argsort(dist)
        match_l = [(idx,dist[idx]) for idx in match_idx[:n] if dist[idx]<=threshold]
        return match_l

    def similarity(self, v1, v2):
        """Find distance between sparse vectors."""
        return sp.sparse.linalg.norm(v1-v2)

