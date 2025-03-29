import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import os
import sys

# This function helps with plotting brackets for heatmaps.
# It is modified from the code at
# https://stackoverflow.com/questions/67235301/vertical-
# grouping-of-labels-with-brackets-on-matplotlib

def make_brackets(
        ax,
        is_horiz,
        pos1, # top/left
        pos2, # bottom/right
        label,
        spine_pos=-0.05,
        tip_pos=-0.01
        ):
    transform = None
    if is_horiz:
        transform = ax.get_xaxis_transform()
        bracket = mpatches.PathPatch(
                mpath.Path(
                    [
                        [pos1,tip_pos],
                        [pos1,spine_pos],
                        [pos2,spine_pos],
                        [pos2,tip_pos]
                        ]
                    ),
                transform=transform,
                clip_on=False,
                facecolor='none',
                edgecolor='k',
                linewidth=2
                )
        ax.add_artist(bracket)
        txt = ax.text(
                (pos1+pos2)/2,
                spine_pos,
                label,
                ha='center',
                va='bottom',
                rotation='horizontal',
                clip_on=False,
                transform=transform
                )
    else:
        transform = ax.get_yaxis_transform()
        bracket = mpatches.PathPatch(
                mpath.Path(
                    [
                        [tip_pos,pos1],
                        [spine_pos,pos1],
                        [spine_pos,pos2],
                        [tip_pos,pos2]
                        ]
                    ),
                transform=transform,
                clip_on=False,
                facecolor='none',
                edgecolor='k',
                linewidth=2
                )
        ax.add_artist(bracket)
        txt = ax.text(
                spine_pos,
                (pos1+pos2)/2,
                label,
                ha='right',
                va='center',
                rotation='vertical',
                clip_on=False,
                transform=transform
                )
    return bracket,txt


