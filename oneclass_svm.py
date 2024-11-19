#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.svm import OneClassSVM

class oneclass_svm:
    def __init__(self, nu_value, kernel = 'rbf', verbose=True):
        self.model = OneClassSVM(nu=nu_value, kernel= kernel, gamma = 'scale', verbose=verbose)            