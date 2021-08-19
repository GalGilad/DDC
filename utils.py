import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties

fp = FontProperties(family="Arial", weight="bold")
globscale = 1.35
CHARS = {"T": TextPath((-0.305, 0), "T", size=1, prop=fp),
         "G": TextPath((-0.384, 0), "G", size=1, prop=fp),
         "A": TextPath((-0.35, 0), "A", size=1, prop=fp),
         "C": TextPath((-0.366, 0), "C", size=1, prop=fp),
         "N": TextPath((-0.325, 0), "N", size=1, prop=fp),
         ">": TextPath((-0.316, 0), ">", size=1, prop=fp),
         "x": TextPath((-0.316, 0), "x", size=1, prop=fp)}
CHAR_TO_COLOR = {'G': 'green',
                 'A': 'blue',
                 'C': 'red',
                 'T': 'orange',
                 'N': 'black',
                 '>': 'black',
                 'x': 'black'}


def draw(c, x, y, yscale=1, ax=None):
    text = CHARS[c]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x, y) + ax.transData
    p = PathPatch(text, lw=0, fc=CHAR_TO_COLOR[c], transform=t)
    if ax != None:
        ax.add_artist(p)
    return p
