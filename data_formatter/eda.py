import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def dataframe_analysis(df: pd.DataFrame):
    precip_cols = ['event_day_acc', 'last_week_acc', 'last_month_acc', 'last_three_months_acc']
    slope_cols = [c for c in df.columns if c.startswith('slope_')]
    elev_cols = [c for c in df.columns if c.startswith('elev_')]

    plt.figure()
    sns.heatmap(df[precip_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Acc precipitation correlation')
    plt.tight_layout()
    plt.savefig('results/precipitation_correlation.png')

    plt.figure()
    sns.heatmap(df[slope_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Slope correlation')
    plt.tight_layout()
    plt.savefig('results/slope_correlation.png')

    plt.figure()
    sns.heatmap(df[elev_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Elevation correlation')
    plt.tight_layout()
    plt.savefig('results/elevation_correlation.png')

    corr_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ['POINT_X', 'POINT_Y']]
    corr_with_target = df[corr_cols].corr()['is_rockfall'].sort_values(ascending=False)
    print(corr_with_target)

    print(df.groupby('hardness')['is_rockfall'].agg(['mean', 'count']))
    print(df.groupby('susceptibility')['is_rockfall'].agg(['mean', 'count']))

    print(pd.crosstab(df['hillside'], df['is_rockfall'], normalize='index'))
