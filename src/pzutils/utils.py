from astropy.table import Table, vstack
import astropy.units as u
from astropy.io import ascii
from astropy.io.ascii import InconsistentTableError
import numpy as np
import scipy as sp

from herschelhelp_internal.utils import mag_to_flux
from herschelhelp.filters import correct_galactic_extinction
from herschelhelp.external import convert_table_for_cigale

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc("figure", figsize=(10, 6))
plt.rcParams.update({"font.size": 14})
import time


def openLephare(filename):
    """Open a LePhare output text file

    The LePhare output text file seems to cause astropy difficulties. It
    also frequently has mismatching headers and columns.

    A mismatch between the column numbers and names is caused by?

    Inputs
    ======

    filename: str
        path to file (relative or absolute)

    Returns
    =======

    table: astropy.table.Table
        The output table in astropy format

    """
    try:
        file1 = open(filename, "r")
        lines = file1.readlines()

        header_line = 0
        # Strips the newline character
        for line in lines:
            if not line.startswith("#"):
                break
            header_line += 1
        header_line -= 1
        # print(header_line)
        table = Table.read(
            filename,
            format="ascii.commented_header",
            header_start=header_line,
            data_start=header_line + 1,
            delimiter="\s",
        )
        # table=ascii.read(filename,format="commented_header")
        return table

    except InconsistentTableError:
        # Some LePhare output tables have a string input from the input catalogue. This
        # needs to be dealt with by adding empty values to various lines and adding
        # columns to the header.
        lines_copy = lines.copy()
        # Count no of cols in every line
        headers_len = (
            len(lines[header_line].split()) - 1
        )  # One hash at start needs to be removed
        line_nos = set()
        for line in lines_copy[header_line + 1 :]:
            line_nos = line_nos.union(set([len(line.split())]))

        # This just adds empty col names
        # additional_headers=['additional_input_col_{}'.format(i) for i in np.arange(np.max(list(line_nos))-headers_len)]

        # This gets the column names from the input catalogue
        in_filename = lines[4].split()[3]  # Get in put cat
        try:
            in_file = open(in_filename, "r")
            in_lines = in_file.readlines()
            string_input = []
            after_zspec = False
            for c in in_lines[0].split():
                if after_zspec:
                    string_input += [c]
                if c == "zspec":
                    after_zspec = True
            additional_headers = string_input + [
                "additional_input_col_{}".format(i)
                for i in np.arange(
                    np.max(list(line_nos)) - (headers_len + len(string_input))
                )
            ]
        except FileNotFoundError:
            additional_headers = [
                "additional_input_col_{}".format(i)
                for i in np.arange(np.max(list(line_nos)) - headers_len)
            ]

        lines_copy[header_line] = (
            lines_copy[header_line][:-1] + " ".join(additional_headers) + "\n"
        )
        # Second loop to append lines missing columns
        for n, line in enumerate(lines_copy):
            if n < header_line + 1:
                continue
            line_len = len(line.split())
            if line_len != np.max(list(line_nos)):
                # add empty cols on lines without extra cols
                lines_copy[n] = (
                    line[:-1]
                    + " ".join(
                        ["-99" for i in np.arange(np.max(list(line_nos)) - line_len)]
                    )
                    + "\n"
                )

        table = Table.read(
            lines_copy,
            format="ascii.commented_header",
            header_start=header_line,
            data_start=header_line + 1,
            delimiter="\s",
        )

        return table


def isMagCol(col):
    """Very simple function to return True if the col is mag col

    based on some manual catalogue checks the mag cols have a band name at the end so
    I can just loop through some standard band names

    """
    bandnames = "ugrizyZYJHK"
    is_mag = False

    # Has standard band name
    for b in bandnames:
        if b == "K":
            b = "Ks"

        if col.endswith("_" + b):
            is_mag = True

    # Wise mags called e.g. W1
    if col.startswith("W") and len(col) == 2:
        is_mag = True

    # Remove distinct VHS fluxes
    if "VHS" in col:
        is_mag = False

    if "NWAY" in col:
        is_mag = False
    # Don't use Gaia mags
    # if col.endswith('_mag'):
    #     is_mag=True
    return is_mag


# def photoz_plots(
#     z1,
#     z1_name,
#     z2,
#     z2_name,
#     xlim=[0, 4],
#     ylim=[0, 4],
#     plot_del=False,
#     name="",
#     log=False,
# ):
#     """Standard photoz compariosn plots

#     Make a grid of direct comparison and bias comparison
#     Maybe also return metircs.
#     """

#     # orig_map=plt.cm.get_cmap('magma')
#     orig_map = plt.colormaps["magma"]
#     # reversed_map=orig_map.reversed()
#     cmap = plt.cm.viridis

#     if plot_del:
#         fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
#             nrows=2, ncols=2, sharey="row", sharex=True, figsize=(10, 10)
#         )

#         # fig.suptitle(name)
#         ax0.set_title("$N$")
#         # ax0.set_aspect('equal')
#         ax1.set_title("$log(N)$")
#         # ax1.set_aspect('equal')
#         # ax2.set_aspect('equal')
#         # ax3.set_aspect('equal')

#         ax0.set_aspect(1)
#         ax1.set_aspect(1)
#         # ax2.set_aspect('equal')
#         # ax3.set_aspect('equal')

#     else:
#         fig, (ax0, ax1) = plt.subplots(ncols=2, sharey="row")  # ,figsize=(10,4))
#         # fig.suptitle(name)
#         ax0.set_title("$N$")
#         ax0.set_aspect("equal")
#         ax1.set_title("$log(N)$")
#         ax1.set_aspect("equal")

#     mask = z1 > xlim[0]
#     mask &= z1 < xlim[1]
#     mask &= z2 > ylim[0]
#     mask &= z2 < ylim[1]
#     delz = (z2 - z1) / (1 + z1)

#     delz_extent = 0.4  # (xlim[1]-xlim[0])
#     delz_lims = [
#         -delz_extent / 2,
#         delz_extent / 2,
#     ]  # [-(xlim[1]-xlim[0])/2,(xlim[1]-xlim[0])/2]
#     mask2 = z1 > xlim[0]
#     mask2 &= z1 < xlim[1]
#     mask2 &= z2 > ylim[0]
#     mask2 &= z2 < ylim[1]
#     mask2 &= delz > delz_lims[0]
#     mask2 &= delz < delz_lims[1]

#     maskPos = (z1 > 0) & (z2 > 0)
#     outlier_frac = np.sum(np.abs(delz[maskPos]) > 0.15) / np.sum(maskPos)
#     sigma_nmad = 1.48 * np.median(np.abs(delz[maskPos]))
#     # print("""Objects in range: {}
#     # Outlier fraction: {}%
#     # Sigma (NMAD):{}
#     # """.format(np.sum(mask),round(outlier_frac/100,2),round(sigma_nmad,3)))

#     fig.suptitle(
#         "{}, {}/{} sources, $\eta$: {}%, $\sigma_{{NMAD}}$: {}".format(
#             name,
#             np.sum(mask),
#             len(mask),
#             round(outlier_frac * 100, 2),
#             round(sigma_nmad, 3),
#         )
#     )

#     n_hex = 30
#     x_scale = np.max(z1[mask]) / xlim[1]
#     y_scale = np.max(z2[mask]) / ylim[1]
#     del_scale = (np.max(delz[mask2]) - np.min(delz[mask2])) / (
#         delz_lims[1] - delz_lims[0]
#     )

#     gridsize = [round(np.sqrt(3) * n_hex * x_scale), round(n_hex * y_scale)]
#     del_gridsize = [round(np.sqrt(3) * n_hex * x_scale), round(n_hex * del_scale)]
#     # ax2.set_aspect(del_scale)
#     # ax3.set_aspect(del_scale)
#     # fig.subplots_adjust(wspace=0,hspace=0)

#     hb0 = ax0.hexbin(
#         z1[mask], z2[mask], cmap=cmap, mincnt=1, linewidths=0.01, gridsize=gridsize
#     )
#     ax0.set(xlim=xlim, ylim=ylim)

#     # ax0.set_xlabel(z1_name)
#     ax0.set_ylabel(z2_name)
#     # ax0.set_xticks([])
#     # ax0.set_title("z phot vs z spec")
#     ax0.plot([0, xlim[1]], [0, xlim[1]], linestyle="-", color="red", linewidth="1.")
#     ax0.plot(
#         [0, xlim[1]], [0.05, 1.05 * xlim[1]], linestyle="-", color="red", linewidth="1."
#     )
#     ax0.plot(
#         [0, xlim[1]],
#         [-0.05, 0.95 * xlim[1]],
#         linestyle="-",
#         color="red",
#         linewidth="1.",
#     )
#     ax0.plot(
#         [0, xlim[1]],
#         [0.15, 1.15 * xlim[1]],
#         linestyle="--",
#         color="red",
#         linewidth="1.",
#     )
#     ax0.plot(
#         [0, xlim[1]],
#         [-0.15, 0.85 * xlim[1]],
#         linestyle="--",
#         color="red",
#         linewidth="1.",
#     )
#     # divider0 = make_axes_locatable(ax0)
#     # cax0 = divider0.append_axes("right", size="5%", pad=0.05)
#     # cb=fig.colorbar(hb0,ax=ax0,cax=cax0)#,label='$N_{Objects}$')

#     hb1 = ax1.hexbin(
#         z1[mask],
#         z2[mask],
#         bins="log",
#         cmap=cmap,
#         mincnt=1,
#         linewidths=0.01,
#         gridsize=gridsize,
#     )
#     ax1.set(xlim=xlim, ylim=ylim)
#     # ax1.set_xlabel(z1_name)
#     # ax1.set_ylabel(z2_name)
#     # ax1.set_title("z phot vs z spec with log colour scale")
#     # ax1.set_xticks([])
#     ax1.plot([0, xlim[1]], [0, xlim[1]], linestyle="-", color="red", linewidth="1.")
#     ax1.plot(
#         [0, xlim[1]], [0.05, 1.05 * xlim[1]], linestyle="-", color="red", linewidth="1."
#     )
#     ax1.plot(
#         [0, xlim[1]],
#         [-0.05, 0.95 * xlim[1]],
#         linestyle="-",
#         color="red",
#         linewidth="1.",
#     )
#     ax1.plot(
#         [0, xlim[1]],
#         [0.15, 1.15 * xlim[1]],
#         linestyle="--",
#         color="red",
#         linewidth="1.",
#     )
#     ax1.plot(
#         [0, xlim[1]],
#         [-0.15, 0.85 * xlim[1]],
#         linestyle="--",
#         color="red",
#         linewidth="1.",
#     )
#     # divider1 = make_axes_locatable(ax1)
#     # cax1 = divider1.append_axes("right", size="5%", pad=0.05)
#     # cb=fig.colorbar(hb1,ax=ax1,cax=cax1)#,label='$log_{10}(N_{Objects})$')

#     if plot_del:
#         hb2 = ax2.hexbin(
#             z1[mask2],
#             delz[mask2],
#             cmap=cmap,
#             mincnt=1,
#             linewidths=0.01,
#             gridsize=del_gridsize,
#         )
#         ax2.set(xlim=xlim, ylim=delz_lims)

#         ax2.set_xlabel(z1_name)
#         ax2.set_ylabel("$\Delta z/(1+z)$")
#         # ax0.set_title("z phot vs z spec")
#         ax2.plot([0, xlim[1]], [0, 0], linestyle="-", color="red", linewidth="1.")
#         ax2.plot(
#             [0, xlim[1]], [-0.05, -0.05], linestyle="-", color="red", linewidth="1."
#         )
#         ax2.plot([0, xlim[1]], [0.05, 0.05], linestyle="-", color="red", linewidth="1.")
#         ax2.plot(
#             [0, xlim[1]], [-0.15, -0.15], linestyle="--", color="red", linewidth="1."
#         )
#         ax2.plot(
#             [0, xlim[1]], [0.15, 0.15], linestyle="--", color="red", linewidth="1."
#         )
#         # divider2 = make_axes_locatable(ax2)
#         # cax2 = divider2.append_axes("right", size="5%", pad=0.05)
#         # cb=fig.colorbar(hb2,ax=ax2,cax=cax2)#,label='$N_{Objects}$')

#         hb3 = ax3.hexbin(
#             z1[mask2],
#             delz[mask2],
#             bins="log",
#             cmap=cmap,
#             mincnt=1,
#             linewidths=0.01,
#             gridsize=del_gridsize,
#         )
#         ax3.set(xlim=xlim, ylim=delz_lims)
#         ax3.set_xlabel(z1_name)
#         # ax3.set_ylabel('$(z_2-z_1)/z_1$')
#         # ax1.set_title("z phot vs z spec with log colour scale")
#         ax3.plot([0, xlim[1]], [0, 0], linestyle="-", color="red", linewidth="1.")
#         ax3.plot(
#             [0, xlim[1]], [-0.05, -0.05], linestyle="-", color="red", linewidth="1."
#         )
#         ax3.plot([0, xlim[1]], [0.05, 0.05], linestyle="-", color="red", linewidth="1.")
#         ax3.plot(
#             [0, xlim[1]], [-0.15, -0.15], linestyle="--", color="red", linewidth="1."
#         )
#         ax3.plot(
#             [0, xlim[1]], [0.15, 0.15], linestyle="--", color="red", linewidth="1."
#         )
#         # divider3 = make_axes_locatable(ax3)
#         # cax3 = divider3.append_axes("right", size="5%", pad=0.05)
#         # cb=fig.colorbar(hb3,ax=ax3,cax=cax3)#,label='$log_{10}(N_{Objects})$')
#         # for ax in [ax0,ax1,ax2,ax3]:
#         #     ax.label_outer()

#     fig_name = (
#         "LePhare_{}_{}_{}".format(z1_name, z2_name, name)
#         .replace("$", "")
#         .replace("{", "")
#         .replace("}", "")
#     )
#     fig.savefig("./figs/{}.png".format(fig_name), bbox_inches="tight")
#     fig.savefig("./figs/{}.pdf".format(fig_name), bbox_inches="tight")
#     return None


def photoz_plots_onecol(
    z1,
    z1_name,
    z2,
    z2_name,
    xlim=[0, 4],
    ylim=[0, 4],
    delz_extent=0.4,
    name="",
    log=False,
    savefig=False,
    plot_medians=False,
    med_bins=30,
    ticks0=None,
    ticks1=None,
    no_plots=False,
):
    """Standard photoz compariosn plots

    Make a grid of direct comparison and bias comparison
    Maybe also return metircs.
    """
    z1 = np.array(z1)
    z2 = np.array(z2)
    # orig_map=plt.cm.get_cmap('magma')
    orig_map = plt.colormaps["magma"]
    # reversed_map=orig_map.reversed()
    cmap = plt.cm.viridis
    stats = {}

    fig, ((ax0), (ax1)) = plt.subplots(
        nrows=2, sharex=True, figsize=(5.4, 10)
    )  # , layout='constrained')

    # fig.suptitle(name)
    # ax0.set_title('$N$')
    # ax0.set_aspect('equal')
    # ax1.set_title('$log(N)$')
    # ax1.set_aspect('equal')
    # ax2.set_aspect('equal')
    # ax3.set_aspect('equal')

    # ax0.set_aspect(1)

    # ax2.set_aspect('equal')
    # ax3.set_aspect('equal')

    mask = z1 > xlim[0]
    mask &= z1 < xlim[1]
    mask &= z2 > ylim[0]
    mask &= z2 < ylim[1]
    delz = (z2 - z1) / (1 + z1)

    # (xlim[1]-xlim[0])
    delz_lims = [
        -delz_extent / 2,
        delz_extent / 2,
    ]  # [-(xlim[1]-xlim[0])/2,(xlim[1]-xlim[0])/2]
    mask2 = z1 > xlim[0]
    mask2 &= z1 < xlim[1]
    # mask2 &= z2 > ylim[0]
    # mask2 &= z2 < ylim[1]
    mask2 &= delz > delz_lims[0]
    mask2 &= delz < delz_lims[1]

    maskPos = (z1 > 0) & (z2 > 0)
    outlier_frac = np.sum(np.abs(delz[maskPos]) > 0.15) / np.sum(maskPos)
    sigma_nmad = 1.48 * np.nanmedian(np.abs(delz[maskPos]))
    bias = np.nanmedian(delz)
    stats["bias"] = bias
    stats["sigma_nmad"] = sigma_nmad
    stats["outlier_frac"] = outlier_frac
    if no_plots:
        return stats
    # print("""Objects in range: {}
    # Outlier fraction: {}%
    # Sigma (NMAD):{}
    # """.format(np.sum(mask),round(outlier_frac/100,2),round(sigma_nmad,3)))

    n_hex = 30
    x_scale = np.max(z1[mask]) / xlim[1]
    y_scale = np.max(z2[mask]) / ylim[1]
    x_scale_del = np.max(z1[mask2]) / xlim[1]
    y_scale_del = (np.max(delz[mask2]) - np.min(delz[mask2])) / (
        delz_lims[1] - delz_lims[0]
    )

    gridsize = [round(np.sqrt(3) * n_hex * x_scale), round(n_hex * y_scale)]
    del_gridsize = [round(np.sqrt(3) * n_hex * x_scale_del), round(n_hex * y_scale_del)]
    # ax1.set_aspect('equal') # 4/0.4

    hb0 = ax0.hexbin(
        z1[mask],
        z2[mask],
        cmap=cmap,
        mincnt=1,
        bins="log",
        linewidths=0.01,
        gridsize=gridsize,
    )
    ax0_maxoc = np.max(hb0.get_array())
    if ticks0 is None:
        ticks0 = axis_scale(ax0_maxoc)
    ax0.set(xlim=xlim, ylim=ylim)

    # ax0.set_xlabel(z1_name)
    ax0.set_ylabel(z2_name)
    # ax0.set_xticks([])
    # ax0.set_title("z phot vs z spec")
    ax0.plot([0, xlim[1]], [0, xlim[1]], linestyle="-", color="red", linewidth="1.")
    ax0.plot(
        [0, xlim[1]],
        [0.05, 0.05 + 1.05 * xlim[1]],
        linestyle="-",
        color="red",
        linewidth="1.",
    )
    ax0.plot(
        [0, xlim[1]],
        [-0.05, -0.05 + 0.95 * xlim[1]],
        linestyle="-",
        color="red",
        linewidth="1.",
    )
    ax0.plot(
        [0, xlim[1]],
        [0.15, 0.15 + 1.15 * xlim[1]],
        linestyle="--",
        color="red",
        linewidth="1.",
    )
    ax0.plot(
        [0, xlim[1]],
        [-0.15, -0.15 + 0.85 * xlim[1]],
        linestyle="--",
        color="red",
        linewidth="1.",
    )
    # divider0 = make_axes_locatable(ax0)
    # cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    # cb=fig.colorbar() #(hb0,ax=ax0,cax=cax0)#,label='$N_{Objects}$')

    # Text description
    textstr = "{}, {}/{} sources visible, \n $\eta$: {:.1f}%, $\sigma_{{NMAD}}$: {:.3f}, bias: {:.3f}".format(
        name, np.sum(mask), len(mask), outlier_frac * 100, sigma_nmad, bias
    )
    # Title
    # fig.suptitle(textstr)
    # Box insert
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax0.text(
        0.03,
        0.97,
        textstr,
        transform=ax0.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    ax0.set_yticks(np.arange(ylim[1] + 1))
    # fig.colorbar(hb0)

    # colourbar
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)

    cbar0 = plt.colorbar(hb0, cax=cax0)
    cbar0.set_ticks(ticks0)
    cbar0.set_ticklabels(ticks0)

    # The delz norm plot

    hb1 = ax1.hexbin(
        z1[mask2],
        delz[mask2],
        cmap=cmap,
        mincnt=1,
        bins="log",
        linewidths=0.01,
        gridsize=del_gridsize,
    )
    ax1_maxoc = np.max(hb1.get_array())
    if ticks1 is None:
        ticks1 = axis_scale(ax1_maxoc)
    ax1.set(xlim=xlim, ylim=delz_lims)

    ax1.set_xlabel(z1_name)
    ax1.set_ylabel("$\Delta z/(1+z)$")
    # ax0.set_title("z phot vs z spec")
    ax1.plot([0, xlim[1]], [0, 0], linestyle="-", color="red", linewidth="1.")
    ax1.plot([0, xlim[1]], [-0.05, -0.05], linestyle="-", color="red", linewidth="1.")
    ax1.plot([0, xlim[1]], [0.05, 0.05], linestyle="-", color="red", linewidth="1.")
    ax1.plot([0, xlim[1]], [-0.15, -0.15], linestyle="--", color="red", linewidth="1.")
    ax1.plot([0, xlim[1]], [0.15, 0.15], linestyle="--", color="red", linewidth="1.")
    # divider2 = make_axes_locatable(ax2)
    # cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    # cb=fig.colorbar(hb2,ax=ax2,cax=cax2)#,label='$N_{Objects}$')
    ax1.set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15])

    # Plot the medians, and 1 sigma interval
    if plot_medians:
        bins, mid_points, delz_16, delz_median, delz_84 = binned_stats(
            z1, z2, bins=med_bins
        )
        ax1.plot(mid_points, delz_median, c="r")
        ax1.plot(mid_points, delz_16, c="r", linewidth=0.3)
        ax1.plot(mid_points, delz_84, c="r", linewidth=0.3)
        ax1.fill_between(mid_points, delz_16, delz_84, color="r", alpha=0.3)

    # colourbar
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    cbar1 = plt.colorbar(hb1, cax=cax1)
    cbar1.set_ticks(ticks1)
    cbar1.set_ticklabels(ticks1)

    plt.subplots_adjust(wspace=0, hspace=0)

    fig_name = (
        "ZZplot_{}_{}_{}".format(z1_name, z2_name, name)
        .replace(" ", "")
        .replace("/", "")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
    )

    if savefig:
        fig.savefig("./figs/{}.png".format(fig_name), bbox_inches="tight")
        fig.savefig("./figs/{}.pdf".format(fig_name), bbox_inches="tight")
    return stats


def fill_nan_linear(arr):
    # Find indices where values are NaN
    nans = np.isnan(arr)
    # Use np.interp to linearly interpolate NaN values
    arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
    return arr


def axis_scale(n):
    """ "Return colorbar values for the scale

    Powers of ten up to largest power of ten below n if n>10
    Else all values lower than 10

    """
    n = int(n)
    if n < 0:
        raise ValueError
    if n <= 10:
        scales = np.arange(0, n)
    if n > 10:
        scales = [int(10**i) for i in np.arange(0, np.floor(np.log10(n)) + 1)]
    return scales


def binned_stats(x, y, bins=30):
    # Get medians and 1 sigma for bins in specz space
    n_bins = bins
    delz = (y - x) / (1 + x)
    bins = np.linspace(np.nanmin(x), np.nanmax(x), n_bins)
    delz_median = np.full(len(bins) - 1, np.nan)
    delz_16 = np.full(len(bins) - 1, np.nan)
    delz_84 = np.full(len(bins) - 1, np.nan)
    mid_points = np.array(
        [(bins[n] + bins[n + 1]) / 2 for n in np.arange(len(bins) - 1)]
    )
    after_empty_bin = False
    for n, b in enumerate(bins[:-1]):
        in_bin = x > bins[n]
        in_bin &= x < bins[n + 1]
        try:
            delz_median[n] = np.nanmedian(delz[in_bin])
            delz_16[n] = np.nanpercentile(delz[in_bin], 15.865)
            delz_84[n] = np.nanpercentile(delz[in_bin], 84.135)
        except ValueError:
            delz_median[n] = np.nan
            delz_16[n] = np.nan
            delz_84[n] = np.nan
        if np.sum(in_bin) < 2 or after_empty_bin:  # Get rid of underfilled bins
            delz_median[n] = np.nan
            delz_16[n] = np.nan
            delz_84[n] = np.nan
            after_empty_bin = True

    delz_median = fill_nan_linear(delz_median)
    delz_16 = fill_nan_linear(delz_16)
    delz_84 = fill_nan_linear(delz_84)
    return bins, mid_points, delz_16, delz_median, delz_84


def row_to_gauss(row, n):
    """Make gaussian function for a given row and gaussian"""

    def gauss(x):
        norm = row["GMM_omegas_{}".format(n)]
        mean = row["GMM_means_{}".format(n)]
        sigma = row["GMM_sigmas_{}".format(n)]
        # print(norm,mean,sigma)
        return (
            np.exp(-((x - mean) ** 2) / (2 * sigma**2))
            * norm
            / (sigma * np.sqrt(2 * np.pi))
        )

    return gauss


def row_to_gmm(row, n=0):
    """create Gaussian Mixture Model from parameters"""
    if n == 0:
        # Number of gaussians
        n = (
            np.max(
                [
                    0 if not c.startswith("GMM_sigmas") else int(c.split("_")[-1])
                    for c in row.colnames
                ]
            )
            + 1
        )

    def gmm(x):
        pdf = np.full(len(x), 0.0)
        # for i in np.arange(23,n+1):
        #     pdf+=row_to_gauss(row,i)(x)
        # return pdf
        g = 0
        # for i in [0,1]:
        #     for j in num_g[i]:
        #         for k in np.arange(j):
        #             pdf+=row_to_gauss(row,g)(x) #/(2*len(num_g[i])*j)
        #             g+=1
        for g in np.arange(n):
            pdf += row_to_gauss(row, g)(x)  # /(2*len(num_g[i])*j)
            g += 1
        return pdf

    return gmm


def make_posterior(row, log=False):
    """Take the gaussian parameters and make the posterior on a grid"""
    z1, z2, dz = -3, 7, 0.01
    n_grid = int((z2 - z1) / dz) + 1
    n_grid = 2000
    if log:
        z_grid = np.logspace(np.log10(0.001), np.log10(6), n_grid)
    else:
        z_grid = np.linspace(z1, z2, n_grid)
    gmm = row_to_gmm(row)
    p_grid = gmm(z_grid)

    return z_grid, p_grid


def row_to_gauss_cdf(row, n):
    """Make gaussian cdf for a given row and gaussian"""

    def gauss(x):
        norm = row["GMM_omegas_{}".format(n)]
        mean = row["GMM_means_{}".format(n)]
        sigma = row["GMM_sigmas_{}".format(n)]
        # print(norm,mean,sigma)
        return (norm / 2) * (1 + sp.special.erf((x - mean) / (sigma * np.sqrt(2))))

    return gauss


def row_to_gmm_cdf(row, n=0):
    """create Gaussian Mixture Model CDF from parameters"""
    if n == 0:
        # Number of gaussians
        n = (
            np.max(
                [
                    0 if not c.startswith("GMM_sigmas") else int(c.split("_")[-1])
                    for c in row.colnames
                ]
            )
            + 1
        )

    def gmm_cdf(x):
        cdf = np.full(len(x), 0.0)
        # for i in np.arange(23,n+1):
        #     pdf+=row_to_gauss(row,i)(x)
        # return pdf
        g = 0
        # for i in [0,1]:
        #     for j in num_g[i]:
        #         for k in np.arange(j):
        #             cdf+=row_to_gauss_cdf(row,g)(x) #/(2*len(num_g[i])*j)
        #             g+=1
        for g in np.arange(n):
            cdf += row_to_gauss_cdf(row, g)(x)  # /(2*len(num_g[i])*j)
            g += 1
        return cdf

    return gmm_cdf


def make_cdf(row, log=False):
    """Take the gaussian parameters and make the cdf on a grid"""
    z1, z2, dz = -3, 7, 0.01
    n_grid = int((z2 - z1) / dz) + 1
    n_grid = 2000
    if log:
        z_grid = np.logspace(np.log10(0.001), np.log10(6), n_grid)
    else:
        z_grid = np.linspace(z1, z2, n_grid)
    gmm_cdf = row_to_gmm_cdf(row)
    cdf_grid = gmm_cdf(z_grid)

    return z_grid, cdf_grid


def cdf_at_value(row, x):
    """Take the gaussian parameters and make the cdf at one value"""

    gmm_cdf = row_to_gmm_cdf(row)
    cdf = gmm_cdf([x])

    return cdf[0]


def outlier_min(zgrid, pdf, max=True):
    """Return the value of z that minimises the outlier fraction

    that is assuming the posterior to be correct what is the probability that
    the specz lies in a region that would not make it an outlier where:
    |z_s-z_p|/(1+z_s)>0.1

    integral of pdf between (z_p-0.15)/1.15 and (z_p+0.15)/0.85
    """
    z1 = [(z - 0.15) / 1.15 for z in zgrid]
    z2 = [(z + 0.15) / 0.85 for z in zgrid]
    # idx = (np.abs(array - value)).argmin()
    i_vals = [(np.abs(zgrid - z)).argmin() for z in z1]
    j_vals = [(np.abs(zgrid - z)).argmin() for z in z2]
    outlier_fracs = np.array(
        [
            np.trapz(pdf[i_vals[n] : j_vals[n]], x=zgrid[i_vals[n] : j_vals[n]])
            for n in np.arange(len(zgrid))
        ]
    )

    # Take absolute maximum
    if max:
        om = zgrid[np.argmax(outlier_fracs)]
    else:
        # If fracs has flat peak take middle value
        n_close = np.sum(
            np.isclose(
                outlier_fracs,
                outlier_fracs[np.argmax(outlier_fracs)],
                atol=1e-3,
                rtol=1e-3,
            )
        )
        om = zgrid[
            np.isclose(
                outlier_fracs,
                outlier_fracs[np.argmax(outlier_fracs)],
                atol=1e-3,
                rtol=1e-3,
            )
        ][n_close // 2]

    return om, outlier_fracs


def quantiles(
    zgrid,
    cdf,
    q_values=[
        (1 - 0.997) / 2,
        (1 - 0.68) / 2,
        0.5,
        1 - (1 - 0.68) / 2,
        1 - (1 - 0.997) / 2,
    ],
):
    """Get the upp and lower sigmas"""
    z_values = []
    for q in q_values:
        z_values.append(zgrid[np.argmin(np.abs(q - cdf))])

    return z_values


def make_posteriors(table, spec_col="SPECZ_REDSHIFT", id_col="id"):
    "Take a circlez output and return the posteriors"
    # Only run for specz objects
    # m=(table[spec_col]>0.02)
    m = np.full(len(table), True)

    posteriors = Table()
    # posteriors['ERO_ID']=erass['ERO_ID'][m]
    # posteriors['id']=table[id_col][m]
    posteriors[id_col] = table[id_col][m]
    pdfs = []
    medians = []
    s3ls, s1ls, s0s, s1us, s3us = [], [], [], [], []
    peaks = []
    outlier_mins = []
    norms = []
    pits = []
    specz = []
    n = 0
    n_fail = 0
    for row in table[m]:
        try:
            post = make_posterior(row)
            zgrid, pdf = post[0], post[1]

            # Remove negative density
            # pdf[zgrid<0]=0.
            # normalise
            norm = np.trapz(pdf, x=zgrid)
            # pdf=pdf/norm #Should already be normalised!

            z_spec = row[spec_col]
            if not isinstance(z_spec, float):
                z_spec = -2
            nmax = np.sum(zgrid < z_spec)
            # pit=np.trapz(p[:nmax],zgrid[:nmax])
            post = make_posterior(row)

            # cdf=[np.trapz(pdf[:n],x=zgrid[:n]) for n in np.arange(len(pdf))]
            # cdf=np.array(cdf)

            cdf = make_cdf(row)[1]

            pit = cdf_at_value(row, z_spec)

            s3l, s1l, s0, s1u, s3u = quantiles(zgrid, cdf)
            # Get median from cdf=0.5
            if np.sum(cdf > 0) > 0:
                median = zgrid[cdf > 0.5][0]
            else:
                median = np.nan

            # zvalue at max pdf
            peak = zgrid[np.isclose(pdf, np.max(pdf))][0]

            # outlier min integral of pdf between (z-0.1)(1+z) and (z+0.1)(1+z)
            z_om, fracs = outlier_min(zgrid, pdf, max=False)

        except IndexError:
            # print('{} failed due to index error'.format(row['ERO_ID']))
            # break
            peak, median, z_om, cdf, norm, pit = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
            s3l, s1l, s0, s1u, s3u = np.nan, np.nan, np.nan, np.nan, np.nan
            pdf = np.full(2000, 0)
            z_spec = np.nan
            n_fail += 1

        n += 1
        # if n>10:
        #     break
        pdfs.append(pdf)
        medians.append(median)
        peaks.append(peak)
        norms.append(norm)
        outlier_mins.append(z_om)
        pits.append(pit)
        specz.append(z_spec)
        s3ls.append(s3l)
        s1ls.append(s1l)
        s0s.append(s0)
        s1us.append(s1u)
        s3us.append(s3u)

        # if n>100:
        #     break
    print("{}failed out of {}".format(n_fail, n))
    posteriors["specz"] = np.array(specz)
    posteriors["pdfs"] = np.array(pdfs)
    posteriors["peaks"] = np.array(peaks)
    posteriors["norms"] = np.array(norms)
    posteriors["s3l"] = np.array(s3ls)
    posteriors["s1l"] = np.array(s1ls)
    posteriors["medians"] = np.array(medians)
    posteriors["s1u"] = np.array(s1us)
    posteriors["s3u"] = np.array(s1us)

    posteriors["outlier_mins"] = np.array(outlier_mins)
    posteriors["pits"] = np.array(pits)
    return posteriors
