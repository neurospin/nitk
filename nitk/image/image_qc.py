import argparse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys; sys.path.append('../..')
from nitk.image.img_to_array import img_to_array

def plot_pca(X, df_description):
    # Assume that X has dimension (n_samples, ...)
    pca = PCA(n_components=2)
    # Do the SVD
    pca.fit(X.reshape(len(X), -1))
    # Apply the reduction
    PC = pca.transform(X.reshape(len(X), -1))
    fig, ax = plt.subplots()
    ax.scatter(PC[:, 0], PC[:, 1])
    # Put an annotation on each data point
    for i, participant_id in enumerate(df_description['participant_id']):
        ax.annotate(participant_id, xy=(PC[i, 0], PC[i, 1]), xytext=(4,4), textcoords='offset pixels')

    plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
    plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def compute_mean_correlation(X, df_description):
    # Compute the correlation matrix
    corr = np.corrcoef(X.reshape(len(X), -1))
    # Compute the Z-transformation of the correlation
    F = 0.5 * np.log((1. + corr) / (1. - corr))
    # Compute the mean value for each sample by masking the diagonal
    F_mean = np.fill_diagonal(F, 0).mean(axis=1)
    # Get the index sorted by descending Z-corrected mean correlation values
    sort_idx = np.argsort(F_mean)[::-1]
    # Get the corresponding ID
    participant_ids = df_description[sort_idx]['participant_id']
    return {"participant_ids": participant_ids, "mean_correlation": F_mean[sort_idx]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='A list of .nii files', required=True, nargs='+', type=str)
    parser.add_argument('--mask', help='A list of .nii masks or a single .nii mask', nargs='+', type=str)
    parser.add_argument('-o', '--output', help='The output .pdf and .csv files', nargs=2,
                        default=['output_qc.pdf', 'qc.csv'], type=str)

    options = parser.parse_args()
    img_filenames = options.input
    mask_filenames = options.mask
    output_paths = options.output

    imgs_arr, df, _ = img_to_array(img_filenames)
    plot_pca(imgs_arr, df)





