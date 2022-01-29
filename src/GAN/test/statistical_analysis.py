import matplotlib.pyplot as plt
import numpy as np
import pandas
import scikit_posthocs as sp
from scipy.stats import stats

col_names = np.array(['original_gen_loss',
             'original_discr_loss',
             'original_sum',
             'original_diff',
             'original_time',
             'augmented_gen_loss',
             'augmented_discr_loss',
             'augmented_sum',
             'augmented_diff',
             'augmented_time',
             'replicated_gen_loss',
             'replicated_discr_loss',
             'replicated_sum',
             'replicated_diff',
             'replicated_time',
              'original_FID',
              'augmented_FID',
              'replicated_FID'])

filename = 'consolidated_result_num_epochs_60_da_pipeline__method__TimeChangeStrategy_factor__6_method__FifthNoteAddStrategy_factor__3_method__OctaveChangeStrategy_factor__1_20220110'
data = pandas.read_csv(f'{filename}.csv', sep=';', na_values='.')
data.columns = col_names


# Remove outliers
# z_scores = stats.zscore(data)
# abs_z_scores = np.abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# data = data[filtered_entries]
num_samples = data.shape[0]
print(num_samples)

# plots with original dataset
fig, ax = plt.subplots(2, 2)
ax[0][0].set_title("Generator Losses")
ax[0][1].set_title("Discriminator Losses")
ax[1][0].set_title("Sum of Losses")
ax[1][1].set_title("Abs Difference of Losses")
data.boxplot(column=[col_names[0], col_names[5], col_names[10]], ax=ax[0][0])
data.boxplot(column=[col_names[1], col_names[6], col_names[11]], ax=ax[0][1])
data.boxplot(column=[col_names[2], col_names[7], col_names[12]], ax=ax[1][0])
data.boxplot(column=[col_names[3], col_names[8], col_names[13]], ax=ax[1][1])
plt.show()

plt.title(f"FID (Frechet Inception Distance) for {num_samples} samples")
data.boxplot(column=[col_names[15], col_names[16], col_names[17]])

# plots with original dataset
fig_new, ax = plt.subplots(2, 2)
ax[0][0].set_title("Generator Losses")
ax[0][1].set_title("Discriminator Losses")
ax[1][0].set_title("Sum of Losses")
ax[1][1].set_title("Abs Difference of Losses")
data.boxplot(column=[col_names[5], col_names[10]], ax=ax[0][0])
data.boxplot(column=[col_names[6], col_names[11]], ax=ax[0][1])
data.boxplot(column=[col_names[7], col_names[12]], ax=ax[1][0])
data.boxplot(column=[col_names[8], col_names[13]], ax=ax[1][1])
plt.show()

plt.title(f"FID (Frechet Inception Distance) for {num_samples} samples")
data.boxplot(column=[col_names[16], col_names[17]])
plt.show()

# combine three groups into one array
discr_data = np.array([data[col_names[15]], data[col_names[16]], data[col_names[17]]])

print(f"Original dataset FID:{np.mean(data[col_names[15]])} +- {np.std(data[col_names[15]])}\n"
      f"Augmented dataset FID:{np.mean(data[col_names[16]])} +- {np.std(data[col_names[16]])}\n"
      f"Replicated dataset FID:{np.mean(data[col_names[17]])} +- {np.std(data[col_names[17]])}\n")
# perform Nemenyi post-hoc test
test = sp.posthoc_nemenyi_friedman(discr_data.T)
print(f"p-values of posthoc Nemenyi Test (FID): {col_names[7]}")
print(test)
