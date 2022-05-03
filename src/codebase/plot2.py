import numpy as np
import holoviews as hv
hv.extension('matplotlib')
hv.output(fig='svg')


def plot_histograms(samples, weights, nbins=50, true_value=None, estimate=None, title=None):

    N = samples.shape[0]

    frequencies, edges = np.histogram(samples, bins=nbins, weights=weights)
    plot = hv.Histogram((edges, frequencies), label = 'Samples')
    if true_value is not None:
        y_length = np.linspace(0, np.max(frequencies), 2)
        true_line = hv.Curve((true_value, y_length ),
                    label = 'True Value').options(ylim=(0, 10), linewidth=3, color='red')
        plot = plot * true_line

    if estimate is not None:
        estimate_line = hv.Curve((estimate, y_length ),
                    label = 'Estimate').options(linewidth=3, color='blue')
        plot = plot * estimate_line

    if title is not None:
        plot = plot.relabel(title)

    return plot


def plot_density(samples, true_value=None, estimate=None, title=None):

    plot = hv.Distribution(samples, label = 'Samples')
    if true_value is not None:
        y_length = np.linspace(0, 1, 2)
        true_line = hv.Curve((true_value, y_length ),
                    label = 'True Value').options(linewidth=4, color='red')
        plot = plot * true_line

    if estimate is not None:
        estimate_line = hv.Curve((estimate, y_length ),
                    label = 'Estimate').options(linewidth=4, color='blue')
        plot = plot * estimate_line

    if title is not None:
        plot = plot.relabel(title)

    return plot


def plot_trace(samples, true_value=None, estimate=None, title=None):

    N = samples.shape[0]
    plot = hv.Curve(samples, label = 'Samples')
    if true_value is not None:
        x_length = np.linspace(0, N, 2)
        true_line = hv.Curve((x_length ,true_value),
                    label = 'True Value').options(linewidth=4, color='red')
        plot = plot * true_line

    if estimate is not None:
        estimate_line = hv.Curve((x_length, estimate ),
                    label = 'Estimate').options(linewidth=4, color='blue')
        plot = plot * estimate_line

    if title is not None:
        plot = plot.relabel(title)

    return plot
