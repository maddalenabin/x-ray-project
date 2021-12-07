import numpy as np
from matplotlib import pyplot as plt


def rmap(n, center, boxlength):
    """Computes the radial distance from the origin.

    Args:
        n (int): the output should have dimension (n,n).
        center (tuple): the center (origin) of the wave.
        boxlength (int): box size in units of wavelength.

    Returns:
        rmap (np.ndarray): the radial distance from the center in each pixel.

    """
    cy, cx = center
    norm = boxlength/float(n)
    X, Y = np.meshgrid(np.arange(n)-cy, np.arange(n)-cx)
    radius = norm * np.sqrt(X**2 + Y**2)
    radius[radius==0] = 1e-2
    return radius

def display_2d(data, title="", ax=None, colorbar=True, cb_kws=None, xlabel=None, ylabel=None, **kwargs):
    """Display 2D data.

    Args:
        data (np.ndarray): input data to display.
        title (str): axes title.
        ax (matplotlib.axes.Axes, optional): axes to plot the data on. Default
            `None` creates new figure.
        colorbar (bool, optional): If True (default), show colorbar.
        cb_kws (dict, optional): optional kwargs for colorbar.
        xlabel (str, optional): Optional label of x-axis.
        ylabel (str, optional): Optional label of y-axis.
        kwargs (optional): arguments passed to plt.imshow

    Returns:
        The displayed image.
    """

    # create an axis if none is provided
    if ax is None:
        fig, ax = plt.subplots()

    if cb_kws is None:
        cb_kws = dict(shrink=.8)

    if xlabel is None:
        xlabel = 'x'
    if ylabel is None:
        ylabel = 'y'

    im = ax.imshow(data, **kwargs)
    if colorbar:
        plt.colorbar(im, ax=ax,  **cb_kws)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.minorticks_on()
    ax.tick_params(which='both', top=True, right=True)
    return im


def to_1d(radius, amplitude):
    """Display 2D data.

    Args:
        data (np.ndarray): input data to display.
        title (str): axes title.
        kwargs (optional): arguments passed to plt.imshow

    Returns:
        The displayed image.
    """

    radius = radius.flatten()
    amplitude = amplitude.flatten()

    ind = np.argsort(radius)

    return radius[ind], amplitude[ind]


def make_rectangular_lattice(dimensions, spacing):
    """Create array of lattice vectors.

    Args:
        dimensions: number of lattice sites (Nx, Ny, Nz).
        spacing: lattice spacing in (x,y,z) in Angstrom.

    Returns:
        Array of lattice vectors with shape (3, Nx * Ny * Nz)

    Example:
        make_rectangular_lattice((2,2,2), (1,1,1))

    """

    vecs = [np.arange(0, d*s, s) for d,s in zip(dimensions, spacing)]
    Rn = np.array(np.meshgrid(*vecs)).reshape(3,-1)
    Rn = Rn - Rn.mean(1, keepdims=True)
    return Rn
