# Energy consumption in Germany between 2006 and 2017, plus production of eolic and photovoltaic energy
# Energy data: https://open-power-system-data.org/

import pandas as pd

opsd = pd.read_csv('time_series_DE_daily.csv', index_col='date', parse_dates=True)

# Exploratory analyses
opsd.index
opsd.shape
opsd.dtypes
opsd.info()

# Add columns for year month and weekday_name
opsd['year'] = opsd.index.year
opsd['month'] = opsd.index.month
opsd['weekday'] = opsd.index.weekday
opsd['weekday'].dtypes

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(11, 4)})
opsd.consumption.plot(linewidth=0.8, color='r')
# the energy consumption seems to stay stable over time

opsd.groupby('year').consumption.aggregate(['mean', 'min', 'max']).plot(color=('g', 'r', 'b'))
# min max and mean of consumption are indeed stable over time

opsd.consumption.resample('M').mean().plot(marker='.', color='r')
# but seasonal oscillations of consumption are evident

cols_plot = ['consumption', 'wind_generation', 'solar_generation']
axes = opsd[cols_plot].plot(marker = '.',
                            linestyle = 'None',
                            alpha = 0.8,
                            figsize = (11, 9),
                            color = ('r', 'b', 'y'),
                            subplots = True)
for ax in axes:
    ax.set_ylabel('Daily totals (GWh)')
# seasonal oscillations are visible also in solar and wind energy production
# but the interesting part here is that consumption as two parallel patterns

# Yearly seasonality
fig, axes = plt.subplots(3, 1, figsize=(15,12), sharex=True)
for var, ax, title in zip(['consumption', 'wind_generation', 'solar_generation'], axes, ['Energy consumption', 'Wind generation', 'Solar generation']):
    sns.boxplot(data=opsd, x='month', y=var, ax=ax)
    ax.set_ylabel('GWh')
    ax.set_title(title)
    if ax != axes[-1]:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Months')
# the yearly pattern is clear here but the two parallel patterns of consumption are not explained

# Check pattern for a single year
ax = opsd.loc['2017', 'consumption'].plot(color='r')
ax.set_ylabel('Daily consumption (GWh)')
ax.set_xlabel('Months')
# The yearly pattern is visible at yearly scale, but oscillations occurr within each month

cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

from pandas.api.types import CategoricalDtype

cat_type = CategoricalDtype(categories=cats, ordered=True)
opsd['weekday'] = opsd['weekday'].astype(cat_type)
opsd.groupby('weekday').consumption.mean().plot(color='r', figsize=(12, 5))
# the two patterns noticed above could be due to weekly patterns
# energy consumption drops on weekends

# Weekly seasonality
sns.boxplot(data=opsd, x='weekday_name', y='consumption')

'''Given the yearly patterns of consumption and renewable production,
how much consumption can rely on renewable across seasons?'''

opsd_quarts = opsd.loc[:, ['consumption', 'wind_generation', 'solar_generation']].resample('Q').sum(min_count=85)
opsd_quarts['quarter'] = opsd_quarts.index.to_period('D').strftime('%Y-Q%q')
# opsd_quarts.index
opsd_quarts = opsd_quarts.set_index(opsd_quarts['quarter'])
opsd_quarts = opsd_quarts.append(pd.Series(name='2018-Q1'))
opsd_quarts['Eolic generation/energy consumption'] = opsd_quarts.wind_generation / opsd_quarts.consumption
opsd_quarts['Photovoltaic generation/energy consumption'] = opsd_quarts.solar_generation / opsd_quarts.consumption
opsd_quarts['Eolic+Photovoltaic/energy consumption'] = opsd_quarts['Eolic generation/energy consumption'] + opsd_quarts['Photovoltaic generation/energy consumption']
opsd_quarts.head(3), opsd_quarts.tail(3)

ax = opsd_quarts.loc['2011-Q4':, ('Eolic generation/energy consumption', 'Photovoltaic generation/energy consumption')].plot.bar(color=('b', 'y'), figsize=(15, 8))
opsd_quarts.loc['2011-Q4':, 'Eolic+Photovoltaic/energy consumption'].plot(color='g', linewidth=2.5, ax=ax)
ax.set_title('Eolic and photovoltaic ratio of total electric consumption per year quarter', fontsize=18)
ax.set_ylabel('Eolic and photovoltaic generation/energy consumption')
ax.set_xlabel('Year quarters')
ax.set_ylim(0, 0.4)
plt.xticks(rotation=90)
ax.legend(fontsize=16)
