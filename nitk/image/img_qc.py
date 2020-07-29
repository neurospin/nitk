import argparse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
import nibabel
from nilearn import plotting
import sys; sys.path.append('../..')
from nitk.image.img_to_array import img_to_array
from nitk.image.img_brain_mask import compute_brain_mask

def plot_pca(X, df_description):
    # Assume that X has dimension (n_samples, ...)
    pca = PCA(n_components=2)
    # Do the SVD
    pca.fit(X.reshape(len(X), -1))
    # Apply the reduction
    PC = pca.transform(X.reshape(len(X), -1))
    fig, ax = plt.subplots(figsize=(20, 30))
    ax.scatter(PC[:, 0], PC[:, 1])
    # Put an annotation on each data point
    for i, participant_id in enumerate(df_description['participant_id']):
        ax.annotate(participant_id, xy=(PC[i, 0], PC[i, 1]), xytext=(4,4), textcoords='offset pixels')

    plt.xlabel("PC1 (var=%.2f)" % pca.explained_variance_ratio_[0])
    plt.ylabel("PC2 (var=%.2f)" % pca.explained_variance_ratio_[1])
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("pca.pdf")
    plt.show()

def compute_mean_correlation(X, df_description):
    # Compute the correlation matrix
    corr = np.corrcoef(X.reshape(len(X), -1))
    # Compute the Z-transformation of the correlation
    F = 0.5 * np.log((1. + corr) / (1. - corr))
    # Compute the mean value for each sample by masking the diagonal
    np.fill_diagonal(F, 0)
    F_mean = F.sum(axis=1)/(len(F)-1)
    np.fill_diagonal(F, 1)
    # Get the index sorted by descending Z-corrected mean correlation values
    sort_idx = np.argsort(F_mean)
    # Get the corresponding ID
    participant_ids = df_description['participant_id'][sort_idx]
    Freorder = F[np.ix_(sort_idx, sort_idx)]
    plt.subplots(figsize=(10, 10))
    cmap = sns.color_palette("RdBu_r", 110)
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(Freorder, mask=None, cmap=cmap, vmin=-1, vmax=1, center=0)
    plt.savefig("corr_mat.pdf")
    plt.show()
    cor = pd.DataFrame(dict(participant_id=participant_ids, corr_mean=F_mean[sort_idx]))
    cor = cor.reindex(['participant_id', 'corr_mean'], axis='columns')
    return cor

def pdf_plottings(nii_filenames, mean_corr, output_pdf, limit=None):
    nslices = (121//2, 145//2, 121//2)
    max_range = limit or len(nii_filenames)
    pdf = PdfPages(output_pdf)
    for i, nii_file in list(enumerate(nii_filenames))[:max_range]:
        fig, ax = plt.subplots(figsize=(15, 10))
        nii = nibabel.load(nii_file)
        plotting.plot_anat(nii, figure=fig, axes=ax, dim=-1,
                           title='Subject %s with mean correlation %.3f' % (mean_corr[i][0], mean_corr[i][1]),
                           cut_coords=(-5, 0, 0))
        plt.subplots_adjust(wspace=0, hspace=0, top=0.9, bottom=0.1)
        pdf.savefig()
        plt.close(fig)
    pdf.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='A list of .nii files', required=True, nargs='+', type=str)
    parser.add_argument('--mask', help='A list of .nii masks or a single .nii mask', nargs='+', type=str)
    parser.add_argument('--output_tsv', help='The output path to the .tsv file', nargs=1, default='mean_cor.tsv', type=str)
    parser.add_argument('--output_pdf', help='The output path to the .pdf file', nargs=1, default='nii_plottings.pdf', type=str)
    parser.add_argument('--limit', help='The max number of slice to plot', default=50, type=int)

    options = parser.parse_args()
    img_filenames = options.input
    mask_filenames = options.mask

    imgs_arr, df, ref_img = img_to_array(img_filenames)

    if mask_filenames is None:
        mask_img = compute_brain_mask(imgs_arr, ref_img)
        mask_arr = mask_img.get_data() > 0
        imgs_arr = imgs_arr.squeeze()[:, mask_arr]
    elif len(mask_filenames) == 1:
        mask_img = nibabel.load(mask_filenames[0])
        mask_arr = mask_img.get_data() > 0
        imgs_arr = imgs_arr.squeeze()[:, mask_arr]
    elif len(mask_filenames) > 1:
        assert len(mask_filenames) == len(imgs_arr), "The list of .nii masks must have the same length as the " \
                                                     "list of .nii input files"
        mask_glob = [nibabel.load(mask_filename).get_data()>0 for mask_filename in mask_filenames]
        imgs_arr = imgs_arr.squeeze()[mask_glob]

    plot_pca(imgs_arr, df)
    mean_corr = compute_mean_correlation(imgs_arr, df)
    mean_corr.to_csv(options.output_tsv, index=False, sep='\t')
    mean_corr = mean_corr.values
    nii_filenames_sorted = [df[df['participant_id'].eq(id)].path.values[0] for (id, _) in mean_corr]
    pdf_plottings(nii_filenames_sorted, mean_corr, options.output_pdf, limit=min(options.limit, len(nii_filenames_sorted)))
