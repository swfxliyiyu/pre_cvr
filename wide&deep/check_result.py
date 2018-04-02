#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 21:19:05 2018

@author: sensetime
"""


if __name__ == '__main__':
    file1 = open('./pred_round1_test_input_A.txt', 'r')
    file2 = open('./../data/round1_ijcai_18_test_a_20180301.txt', 'r')
    line1 = file1.readline()
    line2 = file2.readline()
    cnt=0
    while line1:
        if line1.split(" ")[0] != line2.split(" ")[0]:
            print('ours: ' + line1.split(" ")[0] + ', standard: ' + line2.split(" ")[0])
        else:
            cnt += 1
        line1 = file1.readline()
        line2 = file2.readline()
    print(cnt)
    file1.close()
    file2.close()
