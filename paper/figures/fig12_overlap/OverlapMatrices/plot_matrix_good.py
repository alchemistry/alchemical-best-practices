import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 

#A helper function to read the overlap matrix from file
def get_overlap_matrix(filename, n_states):
    fh = open (filename, 'r')
    lines = fh.readlines()
    fh.close()
    count = 0
    matrix = []
    for line in lines:
        if line.startswith('#Overlap'):
            matrix = lines[(count+1):(count+1+n_states)]
            break
        count = count+1 
    for i in range(len(matrix)):
        temp = matrix[i].strip().split(' ')
        float_temp = [float(j) for j in temp]
        matrix[i] = float_temp
    matrix =np.array(matrix)
    return matrix

# load datasets:
overlap_good = get_overlap_matrix('11-lambda-overlap_good.dat', n_states=11)
overlap_bad = get_overlap_matrix('11-lambda-overlap_bad.dat', n_states=11)



#####Plotting the overlap matrices
# seperately because seaborn subplotting is a bit cumbersome with labels etc
sns.set_context("paper", font_scale=1.8) 

plt.figure(figsize=(8,12))

labels = list(map(str, list(np.arange(1, 12, 1))))

sns.heatmap(overlap_good, annot=True, 
    fmt='.2f', 
    linewidths=.5, 
    annot_kws={"size": 12},
    xticklabels=labels,
    yticklabels=labels,

    )
plt.text(13.85, 4.25, "Degree of overlap",rotation=-90)
plt.xlabel(r'$\lambda$ window')
plt.ylabel(r'$\lambda$ window')
plt.yticks(rotation=0)


plt.savefig("overlap_matrix_good.png" ,dpi=350)

plt.show()


