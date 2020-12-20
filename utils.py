import numpy as np
import pandas as pd
from operator import mul    # or mul=lambda x,y:x*y
from fractions import Fraction
import functools 

def extract_year(str):
    return np.int(str.split("-")[0])

def extract_month(str):
    return np.int(str.split("-")[1])

def match_type(row):
    win_game = 0
    for i in range(1,6):
        if row["winnerset"+str(i)] > row["loserset"+str(i)]:
            win_game += 1
    if win_game == 2:
        return "BO3"
    if win_game == 3:
        return "BO5"
    else:
        return "unknown"
    
def set_result(row, set_num):
    if set_num == 1:
        return(str(row["winnerset1"])+"-"+str(row["loserset1"]))
    if set_num == 2:
        return(str(row["winnerset2"])+"-"+str(row["loserset2"]))
    if set_num == 3:
        return(str(row["winnerset3"])+"-"+str(row["loserset3"]))
    if set_num == 4:
        return(str(row["winnerset4"])+"-"+str(row["loserset4"]))
    if set_num == 5:
        return(str(row["winnerset5"])+"-"+str(row["loserset5"]))
    
def binomial(n,k,p): 
    nck = int(functools.reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1))
    return nck*pow(p,k)*pow(1-p,n-k)
    
def get_game_win_prob(row, game_prob_dict):
    valid_set = [i for i in row[['set1','set2','set3','set4','set5']] if i != '0-0']
    return np.mean([game_prob_dict[i] for i in valid_set])

def set_win_prob(w):
    return pow(w,6)*(1 + 6*(1-w) + 21*pow(1-w,2) + 56*pow(1-w,3) + 126*pow(1-w,4) + 252*w*pow(1-w,5) + 504*0.5*pow(1-w,6))

def match_win_prob(s):
    return pow(s,3)*(1 + 3*(1-s) + 6*pow(1-s,2))
    
def reverse_result(row):
    rowdict = {}
    rowdict['player1id'] = row['player1id']
    rowdict['player2id'] = row['player2id']
    rowdict['year'] = row['year']
    rowdict['month'] = row['month']
    rowdict['matchtype'] = row['matchtype']
    rowdict['match_win_prob'] = 1 - row['match_win_prob']
    return pd.Series(rowdict)