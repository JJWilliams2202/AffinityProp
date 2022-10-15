import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
import distance
import pylev

# data import and remove usernames (removes everything prior and including the ':')
df = pd.read_csv("file.csv", header=None, on_bad_lines='skip')
df[0] = df[0].str.split(':', n=1).str.get(-1)
cleandf = df
# remove numbers and symbols and drop empty strings/duplicates to standardise the entries
cleandf[0] = cleandf[0].str.replace('\d+', '', regex = True)
cleandf[0] = cleandf[0].str.replace('^A-Za-z0-9+', '')
cleandf.replace('', np.nan, inplace=True)
cleandf.dropna(inplace=True)
cleandf.drop_duplicates()
# checking for empty rows
print('Empty Rows:', df.isnull().sum())
smalldf = cleandf.loc[0:4000]
smalldf = np.asarray(smalldf)

# Sometimes the dataset won't converge if there is not enough iterations. 
# Increase iterations if you encounter this problem.
lev_similarity = -1*np.array([[pylev.levenshtein(w1,w2) for w1 in smalldf] for w2 in smalldf])
affprop = AffinityPropagation(affinity="precomputed", damping=0.9, max_iter=1000, verbose=True)
affprop.fit(lev_similarity)
print(lev_similarity)

# generates clusters
for cluster_id in np.unique(affprop.labels_):
	exemplar = smalldf[affprop.cluster_centers_indices_[cluster_id]]
	cluster = np.unique(smalldf[np.nonzero(affprop.labels_==cluster_id)])
	cluster_str = ", ".join(cluster)
	print(" - *%s:* %s" % (exemplar, cluster_str))
