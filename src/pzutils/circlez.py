from astropy.table import Table, join, vstack
import numpy as np
import scipy as sp
from matplotlib import pylab as plt


import rail
from rail.core.data import QPHandle
from qp.metrics.pit import PIT
import qp
from .rail_utils import plot_pit_qq, ks_plot  # Local file

from datetime import date

t = date.today()

gmm_max = 91


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


def row_to_gmm(row, n=gmm_max + 1):
    """create Gaussian Mixture Model from parameters"""

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


def row_to_gmm_cdf(row, n=gmm_max + 1):
    """create Gaussian Mixture Model CDF from parameters"""

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


def make_posteriors(table, spec_col="SPECZ_REDSHIFT", id_col="id"):
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

        except:
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
    posteriors["s3u"] = np.array(s3us)

    posteriors["outlier_mins"] = np.array(outlier_mins)
    posteriors["pits"] = np.array(pits)
    return posteriors
