import matplotlib.pyplot as plt
from os.path import join
import os

def use_style(styles):
    if not isinstance(styles, (list, tuple)):
        styles = [styles]
    for style in styles:
        if style == 'sm':
            pth = join(os.getcwd(), 'utils', 'sm.mplstyle')
        else:
            raise ValueError(f"Style {style} not recognized.")
        plt.style.use(pth)