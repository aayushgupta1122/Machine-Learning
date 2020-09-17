# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:26:19 2020

@author: Admin
"""
from matplotlib import pyplot as plt
def preprocessing_viz(df):
    
    df.head()

    df["pet_category"].unique()
    df["breed_category"].unique()
    df["color_type"].unique()
    df["condition"].unique()
    
    plt.scatter("pet_category", "breed_category", data = df)
    plt.scatter("pet_category", "color_type", data = df)
    plt.scatter("pet_category", "height(m)", data = df)
    plt.scatter("pet_category", "condition", data = df)
    plt.scatter("pet_category", "length(m)", data = df)
    plt.scatter("pet_category", "X1", data = df)
    plt.scatter("pet_category", "X2", data = df)