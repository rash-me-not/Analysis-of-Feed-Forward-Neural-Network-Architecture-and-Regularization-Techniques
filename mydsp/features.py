'''
@author: mroch
'''

import sys

import matplotlib.pyplot as plt


from .utils import fixed_len_spectrogram
from .utils import plot_matrix

import numpy as np
import hashlib  # hash functions


def extract_features_from_corpus(files, adv_ms, len_ms, offset_s,
                                 specfmt="dB", pca=None, pca_axes_N=None,
                                 mel_filters_N=12, verbosity=1):
    """extract_features_from_corpus(files, adv_ms, len_ms, offset_s,
        specfmt, pca, pca_axes_N, mel_filters, verbosity)
        
    Return a 2d array of features.  Each row contains a feature vector
    corresponding to one of the filenames passed in files
    
    Spectral features are extracted based on framing parameters advs_ms, len_ms
    and the center +/- offset_s are retained.
    
    Optional arguments:
    
    specfmt = DFTStream format type, e.g. "dB", "Mel", see DFTStream for 
    details

    pca, pca_axes_N
    These spectra are projected into a PCA space of the specified number
    of pca_axes_N using the PCA space contained in object pca which is of
    type dsp.pca.PCA.
    
    mel_filters
    Only has an effect when specfmt="Mel".  Specifies the number of Mel
    filters to use.
    
    verbosity - Provide information about processing when > 0
        
    This method will attempt to read from cached data as opposed to
    computing the features.  If the cache does not exist, it will be
    created.  Note the the cache files are not portable across machine
    architectures.    
    """

    # This part takes a bit of time, check and see if there is a saved
    # version of the features first

    # Build a filename to load/save features.  
    # Note for PCA, we only hash in whether or not PCA
    # is used and the number of pca_axes_N.  If the PCA analysis changes, 
    # you MUST delete the hash manually.  

    md5 = hashlib.md5()
    string = "".join(files)
    md5.update(string.encode('utf-8'))
    hashkey = md5.hexdigest()

    filenamefmt = "-".join(["features",
                            "adv_ms{}",
                            "len_ms{}",
                            "offset_ms{}",
                            "specfmt{}",
                            "pca{}",
                            "filehash{}.np"])

    pca_desc = "NA" if pca is None else "%d" % (pca_axes_N)
    spec_desc = "%s%s" % (specfmt,
                          "%d" % (mel_filters_N) if specfmt == "Mel" else "")

    filename = filenamefmt.format(
        adv_ms, len_ms, offset_s * 1000, spec_desc, pca_desc, hashkey)

    compute_features = False
    try:
        features = np.load(filename+".npy")
    except IOError:
        # We could process these inside this block, but let's not
        # do our processing inside an exception (trickier for debugging)
        compute_features = True

    if compute_features:
        example_list = []
        count = 0
        for f in files:
            [spec, t, f] = fixed_len_spectrogram(
                f, adv_ms, len_ms, offset_s,
                specfmt=specfmt, mel_filters_N=mel_filters_N)

            if verbosity > 1:
                plt.clf()
                plot_matrix(spec.T, t, f)

            if pca is not None:
                spec = pca.transform(spec, pca_axes_N)

            example_list.append(spec.flatten())

            count += 1
            if verbosity > 0 and count % 100 == 0:
                print("%d " % (count), end="")
                sys.stdout.flush()

        if verbosity > 0:
            print()

        # features = np.concatenate(example_list,axis=0)
        features = np.stack(example_list, axis=1)
        features = features.T

        # Cache on secondary storage for quicker computation next time
        np.save(filename,features)


    return features
