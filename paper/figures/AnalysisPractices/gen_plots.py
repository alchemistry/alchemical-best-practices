#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for generating weakly and strongly correlating example plots for 
explaining good practices in data representation in AFE calculations.
"""

import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import pandas as pd 
import scipy
from sklearn.metrics import mean_absolute_error

###############################################
def interpolate_data(data):
	"""
	1-dimensional linear interpolation to set correlated data to range -15<->-5:
	"""

	return np.interp(data, (data.min(), data.max()), (-12, -8))


# generate toy datasets:
num_points = 50

xx = np.array([-0.51, 51.2])
yy = np.array([0.33, 51.6])
means = [xx.mean(), yy.mean()]  
stds = [xx.std() / 3, yy.std() / 3]
corr = 0.85
covs = [
		[stds[0]**2, stds[0]*stds[1]*corr], 
        [stds[0]*stds[1]*corr, stds[1]**2]
        ] 
m = np.random.multivariate_normal(means, covs, num_points).T

# rescale data to a realistic range, also add two outliers:
x_data_raw = interpolate_data(m[0])
x_data = np.append(x_data_raw, [-11, -12])

y_data_raw = interpolate_data(m[1])
y_data = np.append(y_data_raw, [-8.3, -9])

###############################################
# figure plotting:

sns.set(font='sans-serif', style="ticks")
sns.scatterplot(x_data, y_data, linewidth=0)

# plot kcal bounds:

plt.fill_between(
				x=[-17, -3], 
				y2=[-16.5,-2.5],
				y1=[-17.5, -3.5],
				lw=0, 
				zorder=-10,
				alpha=0.5,
				color="darkorange")
# upper bound:
plt.fill_between(
				x=[-17, -3], 
				y2=[-16,-2],
				y1=[-16.5, -2.5],
				lw=0, 
				zorder=-10,
				color="darkorange", 
				alpha=0.2)
# lower bound:
plt.fill_between(
				x=[-17, -3], 
				y2=[-17.5,-3.5],
				lw=0,
				y1=[-18, -4], 
				zorder=-10,
				color="darkorange", 
				alpha=0.2)



# plot error bars (for predicted):
yerr = np.random.uniform(low=0.1, high=0.4, size=(num_points,))
# append high errors for the outliers and plot them in a different colour:
yerr = np.append(yerr, [0.7, 0.9])
plt.scatter(x=-11, y=-8.3, color="crimson")
plt.scatter(x=-12, y=-9, color="crimson")


plt.errorbar(x_data, y_data, 
			yerr=yerr,
			ls="none",
			lw=0.5, 
			capsize=2,
			color="black",
			zorder=-5
			)


# compute statistics on the datapoints and insert via pandas:
# take datasets without the inserted outliers:

r, p_ignore = scipy.stats.pearsonr(x_data_raw, y_data_raw)
r2 = r**2
mue = mean_absolute_error(x_data_raw, y_data_raw)
rho, p_ignore = scipy.stats.spearmanr(x_data_raw, y_data_raw)
tau, p_ignore = scipy.stats.kendalltau(x_data_raw, y_data_raw)

df = pd.DataFrame(np.array([
							[r"R$^2$", round(r2, 2)],
							[r"MUE (kcal$\cdot$mol$^{-1}$)", round(mue, 2)],
							[r"Spearman $\rho$", round(rho, 2)],
							[r"Kendall $\tau$", round(tau, 2)],

							]),
				columns=["Metric", "Value"])
plt.table(cellText=df.values, colLabels=df.columns, 
				bbox=[0.6,0.05, 0.4,0.3], 
				fontsize=6,
				colWidths=[0.75, 0.25],
				zorder=-20)

# some final formatting:
sns.despine()
plt.xlim(-12.9, -7.1)
plt.ylim(plt.xlim())
plt.xlabel(r"Experimental $\Delta$G [kcal$\cdot$mol$^{-1}$]")
plt.ylabel(r"AFE-predicted $\Delta$G [kcal$\cdot$mol$^{-1}$]")
plt.tight_layout()


plt.savefig("data_depictions.png", dpi=300)
plt.show()






