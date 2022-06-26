import numpy as np
import pandas as pd
import warnings; warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

stocks = pd.read_csv('SP500.csv', error_bad_lines=False)

# plot daily max values for google and amazon during the years
fig, ax = plt.subplots(2, sharex=True, figsize=(16,6))
amazon = stocks.loc[stocks['symbol'] == 'AMZN']
google = stocks.loc[stocks['symbol'] == 'GOOGL']
amazon.groupby('date')['high'].sum().plot(ax=ax[0])
google.groupby('date')['high'].sum().plot(ax=ax[1])
ax[0].set_title('Amazon stock during the years')
ax[1].set_title('google stock during the years')
plt.show()