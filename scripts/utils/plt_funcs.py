import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import os
import sys

# This function helps with plotting brackets.
# It is modified from the code at
# https://stackoverflow.com/questions/67235301/vertical-
# grouping-of-labels-with-brackets-on-matplotlib

# 01. Plot brackets
def make_brackets(
        ax,
        is_horiz,
        pos1, # top/left
        pos2, # bottom/right
        label,
        label_color='k',
        spine_pos=-0.05,
        tip_pos=-0.01,
        txt_rot=None,
        txt_space_factor=1
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
                linewidth=1
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
                linewidth=1
                )
        ax.add_artist(bracket)
        if not txt_rot:
            if spine_pos < tip_pos:
                txt_rot = 90
            elif spine_pos > tip_pos:
                txt_rot = 180
            else:
                txt_rot = 0
        horz_a = None
        if spine_pos < tip_pos:
            horz_a = 'right'
        elif spine_pos > tip_pos:
            horz_a = 'left'
        else:
            horz_a = 'center'
        txt_pos = None
        if spine_pos < tip_pos:
            txt_pos = tip_pos
        else:
            txt_pos = spine_pos
        if type(label)==str:
            label = [label]
        if type(label_color)==str:
            label_color = [label_color]*len(label)
        elif type(label_color)==list:
            if len(label_color) > len(label):
                label_color = label_color[:len(label)]
            elif len(label_color) < len(label):
                n_black = len(label) - len(label_color)
                label_color = label_color = ['black']*n_black

        txt = []
        center_pos = (pos1+pos2)/2
        line_spacing = 3*txt_space_factor
        n_half = int(len(label)/2.0)
        # Note that for the axes here, the origin is
        # bottom left
        pos_list = [
                center_pos+(line_spacing*n_half)-(line_spacing)*idx
                for idx,_ in enumerate(label)
                ]
        for label_curr,color_curr,pos_curr in zip(label,label_color,pos_list):
            txt_obj_curr = ax.text(
                    txt_pos,
                    pos_curr,
                    label_curr,
                    color=color_curr,
                    ha=horz_a, #'right',
                    va='center',
                    rotation=txt_rot, #'vertical',
                    clip_on=False,
                    transform=transform
                    )
            txt.append(txt_obj_curr)
    return bracket,txt

# 02. Helper function to set plot
# width, height in inches, from
# https://stackoverflow.com/questions/44970010/
# axes-class-set-explicitly-size-width-height-
# of-axes-in-given-units
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    return figw,figh

# 03. Helper function to plot a heatmap
# 03. Define a helper function to plot a heatmap
# this function was modified from the matplotlib documentation at
# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks([])
    #ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    #ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    #ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #         rotation_mode="anchor")

    # Turn spines off and create white grid.
    #ax.spines[:].set_visible(False)

    #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im #, cbar

