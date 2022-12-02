from imports import *
"""
Right now, the code is mostly written to for kappa_XY x kappa_CMB where
kappa_XY is reconstrcuted from 2 lines X and Y at the same redshift.
For kappa_Null case, we will need XY combination at some other redshift too.
This has to be added in the code. Since in the example we use in the paper,
two redshifts are z=5 and z=6, we assume them to be equal in terms of power spectra.
This should be changed for more accurate results and also for cases when we have
z1 and z2 very different. This will include calculating the power spectra of
the lines as well as their interlopers at 2 different redshifts as well as
interloper bispectrum and trispectrum with the matter field. Although for
X != Y case, there is only secondary bispectrum effect and that is negligible,
so these extra non-Gaussian terms do not need to be calculated.
"""

class Cell_im(object):

    def __init__(self, exp):
        self.exp = exp
        self.name = self.exp['name']
        #  beam fwhm in radians
        # self.fwhm = self.exp['beam'] * (np.pi/180.)/60.
        #  detector sensitivity in muK*rad.
        # self.sensitivity_t = self.exp['noise_t'] * (np.pi/180.)/60.
        # ell limits
        self.lMin = self.exp['lMin']
        self.lMax = self.exp['lMax']
        self.zc = self.exp['zc']

        """
        self.dl_to_cl = lambda l: 2.*np.pi/(l*(l+1.))
        unlensed = np.loadtxt('input/CAMB/qe_lensedCls.dat')
        fac = self.dl_to_cl(unlensed[:, 0])
        unlensed[:, 1] *= fac  # *1e-2
        # """
        # unlensed spectra assumed to be equal to lensed spectra
        unlensed = np.loadtxt('input/cl_%s_%s.txt' % (self.name, self.zc))

        self.unlensedXX = interp1d(unlensed[:, 0], unlensed[:, 1], kind='linear',
                                   bounds_error=False, fill_value=0.)
        self.unlensedYY = interp1d(unlensed[:, 0], unlensed[:, 2], kind='linear',
                                   bounds_error=False, fill_value=0.)
        self.unlensedXY = interp1d(unlensed[:, 0], unlensed[:, 3], kind='linear',
                                   bounds_error=False, fill_value=0.)

        # ###################### interloper power spectra ####################
        halpha = np.loadtxt('input/auto_cl_Ha_0.11.txt')
        ha = interp1d(halpha[:, 0], halpha[:, -1], kind='linear',
                      bounds_error=False, fill_value=0.)
        co43 = np.loadtxt('input/auto_cl_CO43_0.46.txt')
        co = interp1d(co43[:, 0], co43[:, -1], kind='linear',
                      bounds_error=False, fill_value=0.)

        # #################### interloper bispectra ##################
        # """
        bidata1 = np.loadtxt('input/Bl_1h_matter_Ha_Ha_0.11.txt')
        bidata2 = np.loadtxt('input/Bl_1h_matter_CO43_CO43_0.46.txt')

        self.bispecXX = interp1d(bidata1[:, 0], bidata1[:, -1], kind='linear',
                                 bounds_error=False, fill_value=0.)
        self.bispecYY = interp1d(bidata2[:, 0], bidata2[:, -1], kind='linear',
                                 bounds_error=False, fill_value=0.)
        self.bispecXY = interp1d(bidata2[:, 0], 0.*bidata2[:, -1], kind='linear',
                                 bounds_error=False, fill_value=0.)
        """
        hdulist1 = fits.open('input/Bl_1h_scale_dependent_matter_Ha_Ha_0.11.fits')
        ell_1 = hdulist1[0].data
        bidata1 = hdulist1[1].data
        hdulist1.close()

        hdulist2 = fits.open('input/Bl_1h_scale_dependent_matter_CO43_CO43_0.46.fits')
        ell_2 = hdulist2[0].data
        bidata2 = hdulist2[1].data
        hdulist2.close()

        # bidata1 = np.load('input/Bl_1h_matter_scale_dependent_Ha_Ha_0.11.npy')
        # bidata2 = np.load('input/Bl_1h_matter_scale_dependent_CO43_CO43_0.46.npy')
        # ell_bl = bidata1[:, 0]
        self.bispecXX = rgi((ell_1, ell_1, ell_1), bidata1,
                            method='linear', bounds_error=False, fill_value=0.)
        self.bispecYY = rgi((ell_2, ell_2, ell_2), bidata2,
                            method='linear', bounds_error=False, fill_value=0.)
        self.bispecXY = rgi((ell_1, ell_1, ell_1), 0.*bidata1,
                            method='linear', bounds_error=False, fill_value=0.)
        # """
        # #################### interloper trispectra ##################
        dummy = 1.
        tridata1 = np.loadtxt('input/Tl_shot_Ha_0.11.txt')
        tridata2 = np.loadtxt('input/Tl_shot_CO43_0.46.txt')

        tridata11 = np.loadtxt('input/Tl_1h_Ha_0.11.txt')
        tridata22 = np.loadtxt('input/Tl_1h_CO43_0.46.txt')

        # adding only the 1-halo term of the trispectrum. Shot noise is much
        # smaller. tridata1[:, -1] & tridata2[:, -1]+
        self.trispecXX = interp1d(tridata1[:, 0], dummy*(tridata11[:, -1]), kind='linear',
                                  bounds_error=False, fill_value=0.)
        """
        self.trispecYY = interp1d(tridata[:, 0], np.zeros(len(tridata[:, 0])),
                                  kind='linear',
                                  bounds_error=False, fill_value=0.)
        # """
        self.trispecYY = interp1d(tridata2[:, 0], dummy*(tridata22[:, -1]),
                                  kind='linear',
                                  bounds_error=False, fill_value=0.)
        # """
        self.trispecXY = interp1d(tridata1[:, 0], dummy*np.zeros(len(tridata1[:, 0])),
                                  kind='linear',
                                  bounds_error=False, fill_value=0.)

        white = np.loadtxt('input/whitenoise_cl_Lya_CII_5.txt')
        self.whiteXX = white[0]
        self.whiteYY = white[1]
    
        # total lensed : lens+noise
        print('calculating total power spectra')
        self.totalXX = lambda l: self.unlensedXX(l)+ha(l)  #  + self.artificialNoise(l, 30, 5000)  # +self.detectorNoise(l, self.whiteXX)  # + self.detectorNoise(l, self.sensitivity_t)
        self.totalYY = lambda l: self.unlensedYY(l)+co(l)  #  + self.artificialNoise(l, 30, 5000) # +self.detectorNoise(l, self.whiteYY)  # + self.detectorNoise(l, self.sensitivity_p)
        self.totalXY = lambda l: self.unlensedXY(l)

        zc1 = self.zc
        zc2 = 6.  # second redshift for nulling
        zc3 = 1100.  # cmb
        self.alphanull = self.alpha_null(zc1, zc2, zc3)

    def alpha_null(self, z1, z2, z3):
        x1 = cosmo.comoving_distance(z1).value
        x2 = cosmo.comoving_distance(z2).value
        x2 = cosmo.comoving_distance(z3).value

        num = (1./z3) - (1./z1)
        denom = (1./z1) - (1./z2)
        alpha = num/denom
        return alpha

    def detectorNoise(self, l, white):
        res = np.repeat(white, len(l))
        return res

    def artificialNoise(self, l, lmin, lmax):
        """
        for lmaxT != lmaxP, we add artificial noise on the TT power spectra
        for ell > lmaxT. This way both TT and TE or TB and EE are calculated
        to same ell value, however, TT is dominated by this artificial noise
        and thus dies not really contrubute to the signal. This might bias
        the estimator though. So keep this in mind. For now, just for
        calculating the noise, this should be fine.
        """
        noise = np.zeros(len(l))
        idx = np.where((l < lmin) | (l > lmax))[0]
        noise[idx] = 1.e10
        return noise

    def plot_cell(self):
        nolens = np.loadtxt('input/cl_%s_%s_lmin2_lmax50000.txt' % (self.name, self.zc))

        ell = nolens[:, 0]

        noise_XX = np.repeat(self.whiteXX, len(ell))
        noise_YY = np.repeat(self.whiteYY, len(ell))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # """
        ax.plot(ell, self.unlensedXX(ell), 'k', lw=1.5,
                label=r'XX')
        ax.plot(ell, noise_XX, 'k--', lw=1.5,
                label=r'noise XX')

        ax.plot(ell, self.unlensedYY(ell), 'b', lw=1.5,
                label=r'YY')
        ax.plot(ell, noise_YY, 'b--', lw=1.5,
                label=r'noise YY')

        ax.plot(ell, self.unlensedXY(ell), 'r', lw=1.5,
                label=r'XY')
        """
        ax.plot(ell, ell*(ell+1)*self.unlensedBB(ell)/(2*np.pi), 'g', lw=1.5,
                label=r'BB')
        """
        """
        lens2 = np.loadtxt('../CAMB/manu_lenspotentialCls.dat')
        ell2 = lens2[:, 0]
        noise_t = self.detectorNoise(ell2, self.sensitivity_t)
        noise_p = self.detectorNoise(ell2, self.sensitivity_p)
        ax.plot(ell2, lens2[:, 1]/7.4311e12, 'g', label='manu TT')
        ax.plot(ell2, lens2[:, 2]/7.4311e12, 'g', label='manu EE')
        ax.plot(ell2, lens2[:, 4]/7.4311e12, 'y', label='manu TE')
        """
        ax.legend(loc=2, fontsize='8')  # , labelspacing=0.1)
        ax.set_xscale('log')
        ax.set_yscale('log')  # , nonposy='mask')
        # ax.set_ylim((1.e-15, 1e-9))
        ax.set_xlabel(r'$\ell$', fontsize=16)
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell/2\pi$', fontsize=16)
        plt.show()
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(ell2, ell2*(ell2+1)*self.totalTT(ell2)/(2*np.pi), 'k', lw=1.5,
                label=r'TT')
        # ax.plot(ell2, ell2*(ell2+1)*noise_t/(2*np.pi), 'k--', lw=1.5,
        #         label=r'noise TT')
        ax.plot(ell2, ell2*(ell2+1)*self.totalEE(ell2)/(2*np.pi), 'b', lw=1.5,
                label=r'EE')
        # ax.plot(ell, ell2*(ell2+1)*noise_p/(2*np.pi), 'b--', lw=1.5,
        #         label=r'noise EE')
        ax.plot(ell2, ell2*(ell2+1)*self.totalTE(ell2)/(2*np.pi), 'r', lw=1.5,
                label=r'TE')
        """
        """
        ax.plot(ell, ell*(ell+1)*self.unlensedBB(ell)/(2*np.pi), 'g', lw=1.5,
                label=r'BB')
        """
        """
        ax.legend(loc=2, fontsize='8')  # , labelspacing=0.1)
        ax.set_xscale('log')
        ax.set_yscale('log', nonposy='mask')
        # ax.set_ylim((1.e-15, 1e-9))
        ax.set_xlabel(r'$\ell$', fontsize=16)
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell/2\pi$', fontsize=16)
        plt.show()
        """

    def plot_pk(self):
        data = np.loadtxt('input/pk_Lya_CII_5_lmin2_lmax50000.txt')
        ell = data[:, 0]
        k = data[:, 1]

        noise = np.loadtxt('input/whitenoise_pk_Lya_CII_5.txt')
        noise_XX = np.repeat(noise[0], len(k))
        noise_YY = np.repeat(noise[1], len(k))

        # """
        fig = plt.figure(13)
        ax = fig.add_subplot(111)
        ax.plot(k, data[:, 2], 'k', lw=1.5,
                label=r'XX')
        ax.plot(k, noise_XX, 'k--', lw=1.5,
                label=r'noise XX')
        ax.plot(k, data[:, 3], 'b', lw=1.5,
                label=r'YY')
        ax.plot(k, noise_YY, 'b--', lw=1.5,
                label=r'noise YY')
        ax.plot(k, data[:, 4], 'r', lw=1.5,
                label=r'XY')
        ax.legend(loc=2, fontsize='8')  # , labelspacing=0.1)
        ax.set_xscale('log')
        ax.set_yscale('log', nonposy='mask')
        # ax.set_ylim((1.e-15, 1e-9))
        ax.set_xlabel(r'$k \mathrm{(h/Mpc)}$', fontsize=16)
        ax.set_ylabel(r'$P_k \mathrm{(Jy/sr)}^2 \mathrm{(Mpc/h)}^3$', fontsize=16)
        plt.show()
        # """
        """
        fig = plt.figure(14)
        ax = fig.add_subplot(111)
        # print(self.lensedTT(np.array([100., 500., 1000., 2000., 3500., 5000.])))
        # print(self.totalTT(np.array([100., 500., 1000., 2000., 3500., 5000.])))
        ax.plot(k, totalTT(ell2)/(2*np.pi), 'k', lw=1.5,
                label=r'TT')
        # ax.plot(ell2, ell2*(ell2+1)*noise_t/(2*np.pi), 'k--', lw=1.5,
        #         label=r'noise TT')
        ax.plot(ell2, ell2*(ell2+1)*self.totalEE(ell2)/(2*np.pi), 'b', lw=1.5,
                label=r'EE')
        # ax.plot(ell, ell2*(ell2+1)*noise_p/(2*np.pi), 'b--', lw=1.5,
        #         label=r'noise EE')
        ax.plot(ell2, ell2*(ell2+1)*self.totalTE(ell2)/(2*np.pi), 'r', lw=1.5,
                label=r'TE')
        ax.legend(loc=2, fontsize='8')  # , labelspacing=0.1)
        ax.set_xscale('log')
        ax.set_yscale('log', nonposy='mask')
        # ax.set_ylim((1.e-15, 1e-9))
        ax.set_xlabel(r'$\ell$', fontsize=16)
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell/2\pi$', fontsize=16)
        plt.show()
        """
