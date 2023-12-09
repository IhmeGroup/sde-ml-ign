import pandas as pd
import matplotlib.pyplot as plt


def get_metrics(log_path,metric):
    df = pd.read_csv(log_path)
    return df[['epoch',metric]].dropna()

