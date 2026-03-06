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

plt.figure(figsize=(10,10))
sns.set(font='sans-serif', style="ticks")
plt.tick_params(labelsize=18)


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
plt.scatter(x=-11, y=-8.3, color='#1f77b4')
plt.scatter(x=-12, y=-9, color='#1f77b4')


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

def boostrap_statistic(x_data, y_data, n = 10000):
    n_samples = len(x_data)

    all_stats = {'r2': [],
                 'mue':[],
                 'rho':[],
                 'tau':[]}
    for i in range(n):
        if i == 0:
            x_samples = x_data
            y_samples = y_data
        else:
            samples = np.random.choice(range(n_samples), size=n_samples) # sampling with replacement
            x_samples = [x_data[i] for i in samples]
            y_samples = [y_data[i] for i in samples]
        
        r, _ = scipy.stats.pearsonr(x_samples, y_samples)
        all_stats['r2'].append(r**2)
        mue = mean_absolute_error(x_samples, y_samples)
        all_stats['mue'].append(mue)
        rho, _ = scipy.stats.spearmanr(x_samples, y_samples)
        all_stats['rho'].append(rho)
        tau, _ = scipy.stats.kendalltau(x_samples, y_samples)
        all_stats['tau'].append(tau)

    results = {'r2': {},
               'mue':{},
               'rho':{},
               'tau':{}}
    low_frac = 0.05/2.0 # 95% CI
    high_frac = 1.0 - low_frac
    for stat in all_stats.keys():
        results[stat]['real'] = all_stats[stat][0]
        all_stats[stat] = sorted(all_stats[stat])
        results[stat]['mean'] = np.mean(all_stats[stat])
        results[stat]['low'] = all_stats[stat][int(n*low_frac)]
        results[stat]['high'] = all_stats[stat][int(n*high_frac)]
 
    return results

stats = boostrap_statistic(x_data_raw, y_data_raw)

df = pd.DataFrame(np.array([
							[r"R$^2$", f"{stats['r2']['real']:.2f}$^{{{stats['r2']['high']:.2f}}}_{{{stats['r2']['low']:.2f}}}$"],
							["MUE", f"{stats['mue']['real']:.2f}$^{{{stats['mue']['high']:.2f}}}_{{{stats['mue']['low']:.2f}}}$"],
							[r"Spearman $\rho$", f"{stats['rho']['real']:.2f}$^{{{stats['rho']['high']:.2f}}}_{{{stats['rho']['low']:.2f}}}$"],
							[r"Kendall $\tau$", f"{stats['tau']['real']:.2f}$^{{{stats['tau']['high']:.2f}}}_{{{stats['tau']['low']:.2f}}}$"]
							]),
				columns=["Metric", "Value"])

table = plt.table(cellText=df.values, colLabels=df.columns, 
				bbox=[0.55,0.05, 0.45,0.3], 
				colWidths=[0.3, 0.2],
				zorder=-20)

table.auto_set_font_size(False)
table.set_fontsize(25)
table.scale(1.,1.)

# some final formatting:
sns.despine()
plt.xlim(-12.9, -7.1)
plt.ylim(plt.xlim())
plt.xlabel(r"Experimental $\Delta$G [kcal$\cdot$mol$^{-1}$]", fontsize=25)
plt.ylabel(r"Computed $\Delta$G [kcal$\cdot$mol$^{-1}$]", fontsize=25)
plt.tight_layout()


# we need the image as PDF but seaborn has an issue with negative axis ticks in PDF.
# instead, save as PNG and convert to PDF manually.
# (see https://github.com/mwaskom/seaborn/issues/1107)
plt.savefig("Figure.png", dpi=300)

plt.show()
