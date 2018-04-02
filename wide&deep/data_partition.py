#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 20:46:42 2018

@author: sensetime
"""
import random


def partition(fileRead, fileTrain, fileValidation, splitRatio):
    fread = open(fileRead, "r")
    ftrain = open(fileTrain, "a")
    fval = open(fileValidation, "a")
    
    line = fread.readline()
    line = fread.readline()
    ran = random.random()
    while line:
        if ran > splitRatio:
            ftrain.write(line)
        else:
            fval.write(line)
        line = fread.readline()
        ran = random.random()
    fread.close()
    ftrain.close()
    fval.close()

def cntLines(fileName):
    f = open(fileName, 'r')
    print(fileName)
    print(len(f.readlines()))
    
if __name__ == '__main__':
    partition('./round1_ijcai_18_train_20180301.txt', './train.txt', './val.txt', 0.1)
    cntLines('./train.txt')
    cntLines('./val.txt')