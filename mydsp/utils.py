'''
@author: mroch
'''

from .pca import PCA
from .multifileaudioframes import MultiFileAudioFrames
from .dftstream import DFTStream


# Standard Python libraries
import os.path
from datetime import datetime
# Add-on libraries
import numpy as np
import matplotlib.pyplot as plt

import hashlib  # hash functions
from librosa.feature.spectral import melspectrogram
from statsmodels.tsa.x13 import Spec
from .endpointer import Endpointer


def s_to_frame(s, adv_ms):
    """s_to_frame(s, adv_ms) 
    Convert s in seconds to a frame index assuming a frame advance of adv_ms
    """

    return np.int(np.round(s * 1000.0 / adv_ms))


def plot_matrix(matrix, xaxis=None, yaxis=None, xunits='time (s)', yunits='Hz',
                zunits='(dB rel.)'):
    """plot_matrix(matrix, xaxis, yaxis, xunits, yunits
    Plot a matrix.  Label columns and rows with xaxis and yaxis respectively
    Intensity map is labeled zunits.
    Put "" in any units field to prevent plot of axis label
    
    Default values are for an uncalibrated spectrogram and are inappropriate
    if the x and y axes are not provided
    """

    if xaxis is None:
        xaxis = [c for c in range(matrix.shape[1])]
    if yaxis is None:
        yaxis = [r for r in range(matrix.shape[0])]

    # Plot the matrix as a mesh, label axes and add a scale bar for
    # matrix values
    plt.pcolormesh(xaxis, yaxis, matrix)
    plt.xlabel(xunits)
    plt.ylabel(yunits)
    plt.colorbar(label=zunits)
    plt.show()


def spectrogram(files, adv_ms, len_ms, specfmt="dB", mel_filters_N=12):
    """spectrogram(files, adv_ms, len_ms, specfmt)
    Given a filename/list of files and framing parameters (advance, length in ms), 
    compute a spectrogram that spans the files.
    
    Type of spectrogram (specfmt) returned depends on DFTStream, see class
    for valid arguments and interpretation, defaults to returning
    intensity in dB.
    
    Returns [intensity, taxis_s, faxis_Hz]
    """

    # If not a list, make it so number one...
    if not isinstance(files, list):
        files = [files]

    # Set up frame stream and pass to DFT streamer
    framestream = MultiFileAudioFrames(files, adv_ms, len_ms)
    dftstream = DFTStream(framestream, specfmt=specfmt, mels_N=mel_filters_N)

    # Grab the spectra
    spectra = []
    for s in dftstream:
        spectra.append(s)

    # Convert to matrix
    spectra = np.asarray(spectra)

    # Time axis in s
    adv_s = framestream.get_frameadv_ms() / 1000
    t = [s * adv_s for s in range(spectra.shape[0])]

    return [spectra, t, dftstream.get_Hz()]


def fixed_len_spectrogram(file, adv_ms, len_ms, offset_s, specfmt="dB",
                          mel_filters_N=12):
    """fixed_len_spectrogram(file, adv_ms, len_ms, offset_s, specfmt, 
        mel_filters_N)
        
    Generate a spectrogram from the given file.
    Truncate the spectrogram to the specified number of seconds
    
    adv_ms, len_ms - Advance and length of frames in ms
    
    offset_s - The spectrogram will be truncated to a fixed duration,
        centered on the median time of the speech distribution.  The
        amount of time to either side is determned by a duration in seconds,
        offset_s.
        
    specfmt - Spectrogram format. See dsp.dftstream.DFTStream for valid formats
    """

    # Use the Endpointer class to determine the times associated with speech.
    endpoint = Endpointer(file, adv_ms, len_ms)
    [sfull, sfull_t, sfull_in] =  spectrogram(file,adv_ms,len_ms)
    spectra_speech = sfull[endpoint.predictions == endpoint.speech_category]

    speech_index =  np.where(endpoint.speech_frames())
    time_speech = np.asarray(sfull_t)[speech_index]
    median = np.median(speech_index)
    offset_label = offset_s/(adv_ms*0.001)
    start = int(round(median - offset_label))
    end = int(round(median + offset_label))
    s_result = sfull[start:end]

    # Pad the left and right sides with zeros if too short.

    if( end > sfull.shape[0]):
        diff_end = int(end - sfull.shape[0])
        s_result = np.pad(sfull[start:end],((0,diff_end), (0, 0)), 'constant', constant_values='0')
    if( start < 0):
        diff_start = int(0 - start)
        s_result = np.pad(sfull[0:end],((diff_start, 0), (0, 0)), 'constant', constant_values='0')

    t = [s * adv_ms * 0.001 for s in range(s_result.shape[0])]
    return [s_result, t, sfull_in]


def pca_analysis_of_spectra(files, adv_ms, len_ms, offset_s):
    """"pca_analysis_of_spectra(files, advs_ms, len_ms, offset_s)
    Conduct PCA analysis on spectra of the given files
    using the given framing parameters.  Only retain
    central -/+ offset_s of spectra
    """

    md5 = hashlib.md5()
    string = "".join(files)
    md5.update(string.encode('utf-8'))
    hashkey = md5.hexdigest()

    filename = "VarCovar-" + hashkey + ".pcl"
    try:
        pca = PCA.load(filename)

    except FileNotFoundError:
        example_list = []
        for f in files:
            [example, _t, _f] = fixed_len_spectrogram(f, adv_ms, len_ms,
                                                      offset_s, "dB")
            example_list.append(example)

        # row oriented examples
        spectra = np.vstack(example_list)

        # principal components analysis
        pca = PCA(spectra)

        # Save it for next time
        pca.save(filename)

    return pca


def get_corpus(dir, filetype=".wav"):
    """get_corpus(dir, filetype=".wav"
    Traverse a directory's subtree picking up all files of correct type
    """

    files = []

    # Standard traversal with os.walk, see library docs
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(filetype)]:
            files.append(os.path.join(dirpath, filename))

    return files


def get_class(files):
    """get_class(files)
    Given a list of files, extract numeric class labels from the filenames
    """

    # TIDIGITS single digit file specific

    classmap = {'z': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'o': 10}

    # Class name is given by first character of filename    
    classes = []
    for f in files:
        dir, fname = os.path.split(f)  # Access filename without path
        classes.append(classmap[fname[0]])

    return classes


class Timer:
    """Class for timing execution
    Usage:
        t = Timer()
        ... do stuff ...
        print(t.elapsed())  # Time elapsed since timer started        
    """

    def __init__(self):
        "timer() - start timing elapsed wall clock time"
        self.start = datetime.now()

    def reset(self):
        "reset() - reset clock"
        self.start = datetime.now()

    def elapsed(self):
        "elapsed() - return time elapsed since start or last reset"
        return datetime.now() - self.start
