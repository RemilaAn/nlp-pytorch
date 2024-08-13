import pandas as pd


def get_data():
    file_name = 'IMDB Dataset.csv'
    df = pd.read_csv(file_name)
    # df.head()
    return [df['review'].values, df['sentiment'].values]
