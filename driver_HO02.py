import time
# import imp
# import cell_cmb
# imp.reload(cell_cmb)
from cell_im import *
import HO02_QE as ho

time0 = time()

fam = "serif"
plt.rcParams["font.family"] = fam

# """
# first
first = {"name": "Lya_CII", "lMin": 30., "lMax": 1500.,
      "zc": 5.}

im = Cell_im(first)

l_est = ho.lensing_estimator(im)

# est = ['XX']  # , 'YY', 'XY']
est = ['XX', 'YY', 'XY']

print ("Running HO02 estimator")
# l_est.calc_var(est)
l_est.interp_var(est)
# l_est.plot_var(est)

# l_est.calc_trispec(est)
l_est.interp_trispec(est)
# l_est.calc_primbispec(est)
l_est.interp_primbispec(est)
# l_est.calc_secbispec(est)
l_est.interp_secbispec(est)

# l_est.calc_cov(est)
l_est.interp_cov(est)
# l_est.plot_cov(est)
# l_est.plot_cov_XX(['XX'])
# l_est.plot_cov_pair()
# l_est.plot_cov_gaussian_total(est)
# l_est.plot_corrcoef(est)
pairs = ['XX-XX', 'XX-YY', 'XX-XY', 'XY-XY', 'XX-CMB', 'XY-CMB']
# l_est.plot_kappa_pairs(pairs)

fsky = 0.4
# lmaxarray = np.array([300., 600., 1000., 1500.])  # , 2000., 3000., 4000.])
lmaxarray = np.array([600., 1000., 1500., 2000., 3000.])  # , 4000.])

def plt_multlmax_snr(maxlarray):
    lines = ["-", "--", "-."]
    cl = ["g", "b", "r", "c", "k", "m"]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    for i in range(len(maxlarray)):
        first = {"name": "Lya_CII", "lMin": 30., "lMax": maxlarray[i],
                 "zc": 5.}
        im = Cell_im(first)    
        l_est = ho.lensing_estimator(im)
        l_est.interp_var(est)
        l_est.interp_trispec(est)
        l_est.interp_primbispec(est)
        l_est.interp_secbispec(est)
        # l_est.interp_cov(est)

        theta = 180.*60./maxlarray[i]
        lcen, snrbin, snrbin_g, snrnull = l_est.snr_clkappakappa(est, fsky)
        ax.plot(lcen, snrbin, c=cl[i], ls='--', label=r"$\ell_\mathrm{max \: LIM}$ = %d  (%d$^{'}$)" % (maxlarray[i], theta))
        # lcen, snrbin, snrbin_g = l_est.snr_clkappanullkappacmb(est, fsky)
        ax.plot(lcen, snrnull, c=cl[i], ls='-')  # , label=r'lMax = %d' % (maxlarray[i]))

    ax.plot([], [], c='k', ls='-', label=r"$C_L^{\hat{\kappa}_{\rm Null} \hat{\kappa}_{\rm CMB}}$")
    ax.plot([], [], c='k', ls='--', label=r"$C_L^{\hat{\kappa}_{\rm XY} \hat{\kappa}_{\rm CMB}}$")
    ax.legend(fontsize='18', loc='lower right', frameon=False)  # , labelspacing=0.1)
    ax.set_xscale('log')
    ax.set_yscale('log')  # , nonposy='mask')
    ax.set_xlabel(r'$L$', fontsize=24)
    # ax.set_ylabel(r'Cumulative SNR $C_L^{\hat{\kappa}_{\rm XY} \hat{\kappa}_{\rm CMB}}$', fontsize=24)
    ax.set_ylabel(r'Cumulative SNR $C_L^{\hat{\kappa} \hat{\kappa}}$', fontsize=24)
    ax.set_ylim(ymin=0.07, ymax=300)
    ax.set_xlim((5., 3.e3))
    ax.tick_params(axis='both', labelsize=20)
    ax2 = ax.secondary_yaxis("right")
    ax2.tick_params(axis="y", direction="out", labelright=False)
    plt.show()
    # plt.savefig('output/Figures/cumSNR_Cl^kappaXYkappaCMB_diff_lmax_fsky%s.pdf' %(fsky), bbox_inches="tight")
    # plt.savefig('output/Figures/cumSNR_Cl^kappaXYkappaCMB_Cl^kappanullkappaCMB_diff_lmax_fsky%s.pdf' %(fsky), bbox_inches="tight")
    # plt.savefig('output/Figures/cumSNR_Cl^kappaXYkappaCMB_Cl^kappanullkappaCMB_diff_lmax_fsky%s_alphaincluded.pdf' %(fsky), bbox_inches="tight")

plt_multlmax_snr(lmaxarray)

def plt_multlmax_Abias(maxlarray):
    lines = ["-", "--", "-."]
    cl = ["g", "b", "r", "c", "k", "m"]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    for i in range(len(maxlarray)):
        first = {"name": "Lya_CII", "lMin": 30., "lMax": maxlarray[i],
                 "zc": 5.}
        im = Cell_im(first)    
        l_est = ho.lensing_estimator(im)
        l_est.interp_var(est)
        l_est.interp_trispec(est)
        l_est.interp_primbispec(est)
        l_est.interp_secbispec(est)
        # l_est.interp_cov(est)

        lcen, bias_Ahat = l_est.A_bias_XX(fsky)
        ax.plot(lcen, bias_Ahat, c=cl[i], ls='-', label=r'$\ell_\mathrm{max \: LIM}$ = %d' % (maxlarray[i]))
        # ax.plot(lcen, snrbin_g, c=cl[i], ls='--')  # , label=r'lMax = %d' % (maxlarray[i]))

    # ax.plot([], [], c='k', ls='-', label="Total noise bias")
    # ax.plot([], [], c='k', ls='--', label="Gaussian noise bias")
    ax.legend(fontsize='14', loc='upper right', frameon=False)  # , labelspacing=0.1)
    ax.set_xscale('log')
    # ax.set_yscale('log')  # , nonposy='mask')
    ax.set_xlabel(r'$L_{\rm max}$', fontsize=24)
    ax.set_ylabel(r'Bias on $C_L^{\hat{\kappa}_{\rm XX} \hat{\kappa}_{\rm XX}}$', fontsize=24)
    ax.set_ylim(ymax=0.9)
    ax.set_xlim((5., 3.e3))
    ax.tick_params(axis='both', labelsize=20)
    plt.show()
    # plt.savefig('output/Figures/cumbias_Cl^kappaXXkappaXX_diff_lmax_fsky%s_1halotrispecadded.pdf' %(fsky), bbox_inches="tight")

# plt_multlmax_Abias(lmaxarray)

def plt_di_dz():
    z = np.linspace(0, 7, 100)
    y = np.zeros(len(z))
    x_ticks = [0.09, 0.11, 0.46, 0.56, 0.82, 1.56, 1.7]
    xtick_labels = ['0.09', '0.11', '0.46', '0.56', '0.82', '1.56', '5.0']
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(z, y)
    ax.set_xlabel(r'$z$', fontsize=16)
    ax.set_ylabel(r'$dI/dz$', fontsize=16)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(xtick_labels)
    """
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    # ax.yaxis.set_ticks_position('left')
    # ax.spines['left'].set_position(('data',0))
    plt.xticks(x_ticks, xtick_labels, fontsize='small', rotation=45)
    plt.ylim()
    plt.show()

# plt_di_dz()
print(time()-time0)