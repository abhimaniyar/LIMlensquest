from imports import *
"""
Right now, the code is mostly written to for kappa_XY x kappa_CMB where
kappa_XY is reconstrcuted from 2 lines X and Y at the same redshift.
For kappa_Null case, we will need XY combination at some other redshift too.
This has to be added in the code. Since in the example we use in the paper,
two redshifts are z=5 and z=6, we assume them to be equal in terms of power spectra.
This should be changed for more accurate results.
"""


class lensing_estimator(object):

    def __init__(self, Cell_im):
        self.im = Cell_im
        self.name = self.im.name
        # self.beam = self.im.exp['beam']
        # self.noise = self.im.exp['noise_t']

        """
        bounds for ell integrals
        l_1 + l_2 = L
        """
        self.l1Min = self.im.lMin
        # max value for l1 and l2 is taken to be same
        self.l1Max = self.im.lMax

        # L = l_1 + l_2. This L is for reconstructed phi field
        # self.L = np.logspace(np.log10(1.), np.log10(2.*self.l1Max+1.), 51, 10.)
        # a1 = np.logspace(np.log10(1.), np.log10(100.), 20, 10.)
        # a2 = np.logspace(np.log10(110.), np.log10(1500.), 140, 10.)
        # a3 = np.logspace(np.log10(1600.), np.log10(2*self.l1Max+1.), 51, 10.)
        # self.L = np.concatenate((a1, a2, a3))
        self.L = np.logspace(np.log10(2.), np.log10(2*self.l1Max+1.), 51, 10.)
        # self.L = np.logspace(np.log10(2.), np.log10(2*self.l1Max+1.), 101, 10.)
        # self.L = np.linspace(1., 201., 1001)
        self.Nl = len(self.L)
        self.N_phi = 50  # 200  # number of steps for angular integration steps
        # reduce to 50 if you need around 0.6% max accuracy till L = 3000
        # from 200 to 400, there is just 0.03% change in the noise curves till L=3000
        self.var_out = 'output/HO02_variance_individual_%s_lmin%s_lmax%s_Nl%s_Nphi%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax), str(self.Nl), str(self.N_phi))
        self.covar_out = 'output/HO02_covariance_%s_lmin%s_lmax%s_Nl%s_Nphi%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax), str(self.Nl), str(self.N_phi))
        self.primbispec_out = 'output/HO02_primbispec_%s_lmin%s_lmax%s_Nl%s_Nphi%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax), str(self.Nl), str(self.N_phi))
        self.secbispec_out = 'output/HO02_secbispec_%s_lmin%s_lmax%s_Nl%s_Nphi%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax), str(self.Nl), str(self.N_phi))
        self.trispec_out = 'output/HO02_trispec_%s_lmin%s_lmax%s_Nl%s_Nphi%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax), str(self.Nl), str(self.N_phi))
        self.covar_out_ng = 'output/HO02_covariance_nonGaussian_%s_lmin%s_lmax%s_Nl%s_Nphi%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax), str(self.Nl), str(self.N_phi))

    """
    L = l1 + l2
    phi1 = angle betweeen vectors (L, l_1)
    phi2 = angle betweeen vectors (L, l_2)
    and phi12 = phi1 - phi2
    """

    def l2(self, L, l_1, phi1):
        """
        mod of l2 = (L-1_1) given phi1
        """
        return np.sqrt(L**2 + l_1**2 - 2*L*l_1*np.cos(phi1))

    def mag_l(self, l_1, l_2, phi_12):
        """
        mod of l = (vec(l_1) + vec(l_2)) given phi_12 which is the anngle
        between them
        """
        res = l_1**2 + l_2**2 + 2*l_1*l_2*np.cos(phi_12)
        # """
        idx = np.where(res < 0)
        if idx[0].size > 1:
            res[idx[0]] = 0.  # np.abs(res[idx[0]])
        # """
        # res = np.abs(res)
        return res  # np.sqrt(l_1**2 + l_2**2 + 2*l_1*l_2*np.cos(phi_12))

    def phi12(self, L, l_1, phi1):
        """
        phi12 = phi1 - phi2
        """
        x = L*np.cos(-phi1) - l_1
        y = L*np.sin(-phi1)
        result = -np.arctan2(y, x)
        return result

    def phi2(self, L, l_1, phi1):
        """
        phi2 = phi1 - phi12
        """
        result = phi1 - self.phi12(L, l_1, phi1)
        # result = phi1 + self.phi12(L, l_1, phi1)
        return result

    def l4(self, L, l_3, phi3):
        """
        mod of l4 = (L'-1_3) = -L-l_3 given phi3
        """
        return np.sqrt(L**2 + l_3**2 + 2*L*l_3*np.cos(phi3))

    def phi34(self, L, l_3, phi3):
        """
        phi34 = phi3 - phi4
        """
        x = -L*np.cos(phi3) - l_3
        y = L*np.sin(phi3)
        result = -np.arctan2(y, x)
        return result

    def phi4(self, L, l_3, phi3):
        """
        phi4 = phi3 - phi34
        """
        result = phi3 - self.phi34(L, l_3, phi3)
        # result = phi1 + self.phi12(L, l_1, phi1)
        return result

    def f_XY(self, L, l_1, phi1, XY):
        """
        lensing response such that
        <X_l1 Y_{L-l1}> = f_XY(l1, L-l1)*\phi_L.
        For IM, response function is same for XX, YY, and XY and is the same
        as the TT response function of the CMB
        """

        l_2 = self.l2(L, l_1, phi1)
        # phi12 = self.phi12(L, l_1, phi1)
        phi2 = self.phi2(L, l_1, phi1)

        Ldotl_1 = L*l_1*np.cos(phi1)
        Ldotl_2 = L*l_2*np.cos(phi2)
        # """
        if XY == 'XX':
            result = self.im.unlensedXX(l_1)*Ldotl_1
            result += self.im.unlensedXX(l_2)*Ldotl_2
        elif XY == 'YY':
            result = self.im.unlensedYY(l_1)*Ldotl_1
            result += self.im.unlensedYY(l_2)*Ldotl_2
        elif XY == 'XY':
            result = self.im.unlensedXY(l_1)*Ldotl_1
            result += self.im.unlensedXY(l_2)*Ldotl_2
        # result *= 2. / L**2

        return result

    def f_XY_bispec(self, l__1, l__2, phi_12, XY):
        """
        lensing response such that
        <X_l1 Y_{L-l1}> = f_XY(l1, L-l1)*\phi_L.
        Here this is defined for calculating the terms in the secondary
        bispectrum. Here this is defined as
        f_XY(l_1, l_2) = C_(l_1)*[l_1**2 + l_1*l_2*cos(phi12)] + C_(l_2)*[l_2**2 + l_1*l_2*cos(phi12)]
        Not used in the code.
        """

        L12dotl__1 = l__1**2 + l__1*l__2*np.cos(phi_12)
        L12dotl__2 = l__2**2 + l__1*l__2*np.cos(phi_12)
        # """
        if XY == 'XX':
            result = self.im.unlensedXX(l__1)*L12dotl__1
            result += self.im.unlensedXX(l__2)*L12dotl__2
        elif XY == 'YY':
            result = self.im.unlensedYY(l__1)*L12dotl__1
            result += self.im.unlensedYY(l__2)*L12dotl__2
        elif XY == 'XY':
            result = self.im.unlensedXY(l__1)*L12dotl__1
            result += self.im.unlensedXY(l__2)*L12dotl__2

        # result *= 2. / L**2

        return result

    def F_XY(self, L, l_1, phi1, XY):
        """
        Weighing terms for the estimator.
        This decides the weights for a corresponding pair of multipoles for
        X and Y.
        """

        l_2 = self.l2(L, l_1, phi1)
        phi2 = self.phi2(L, l_1, phi1)

        if XY == 'XX':
            numerator = self.f_XY(L, l_1, phi1, XY)
            denominator = 2.*self.im.totalXX(l_1)*self.im.totalXX(l_2)
            result = numerator/denominator
        elif XY == 'YY':
            numerator = self.f_XY(L, l_1, phi1, XY)
            denominator = 2.*self.im.totalYY(l_1)*self.im.totalYY(l_2)
            result = numerator/denominator
        elif XY == 'XY':
            numerator = self.im.totalYY(l_1)*self.im.totalXX(l_2)*self.f_XY(L, l_1, phi1, XY)
            numerator -= self.im.totalXY(l_1)*self.im.totalXY(l_2)*self.f_XY(L, l_2, phi2, XY)
            denominator = (self.im.totalXX(l_1)*self.im.totalYY(l_2) *
                           self.im.totalYY(l_1)*self.im.totalXX(l_2))
            denominator -= (self.im.totalXY(l_1)*self.im.totalXY(l_2))**2
            result = numerator/denominator
        return result

    def F_XY_bispec(self, L, l__1, phi_1, XY):
        """
        Weighing terms for the estimator.
        This decides the weights for a corresponding pair of multipoles for
        X and Y. Not used in the code.
        """

        l_2 = self.l4(L, l__1, phi_1)
        # phi2 = self.phi4(L, l__1, phi_1)
        phi12 = self.phi34(L, l__1, phi_1)  # phi2 - phi_1

        if XY == 'XX':
            numerator = self.f_XY_bispec(l__1, l_2, phi12, XY)
            denominator = 2.*self.im.totalXX(l__1)*self.im.totalXX(l_2)
            result = numerator/denominator
        elif XY == 'YY':
            numerator = self.f_XY_bispec(l__1, l_2, phi12, XY)
            denominator = 2.*self.im.totalYY(l__1)*self.im.totalYY(l_2)
            result = numerator/denominator
        elif XY == 'XY':
            numerator = self.im.totalYY(l__1)*self.im.totalXX(l_2)*self.f_XY_bispec(l__1, l_2, phi12, XY)
            numerator -= self.im.totalXY(l__1)*self.im.totalXY(l_2)*self.f_XY_bispec(l__1, l_2, phi12, XY)
            denominator = (self.im.totalXX(l__1)*self.im.totalYY(l_2) *
                           self.im.totalYY(l__1)*self.im.totalXX(l_2))
            denominator -= (self.im.totalXY(l__1)*self.im.totalXY(l_2))**2
            result = numerator/denominator
        return result

    def var_individual(self, L, XY):
        """
        Variance of the QE for individual XY. Same as the
        normalization of the \phi_XY estimator. Such that norm = 1/(\int F*f)
        such that \phi_XY = norm* \int F*(XY)
        """
        l1min = self.l1Min
        l1max = self.l1Max
        # """
        if L > 2.*l1max:  # L = l1 + l2 thus max L = 2*l1
            return 0.
        # """

        def integrand(l_1, phil):

            l_2 = self.l2(L, l_1, phil)
            """
            if l_1 < l1min or l_2 < l1min or l_1 > l1max or l_2 > l1max:
                return 0.
            """
            result = self.f_XY(L, l_1, phil, XY)*self.F_XY(L, l_1, phil, XY)

            result *= 2*l_1  # **2
            # d^2l_1 = dl_1*l_1*dphi1
            """factor of 2 above because phi integral is symmetric. Thus we've
            put instead of 0 to 2pi, 2 times 0 to pi
            Also, l_1^2 instead of l_1 if we are taking log spacing for
            l_1"""
            result /= (2.*np.pi)**2
            # """
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            # print idx
            result[idx] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            return result


        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        # l1 = np.logspace(np.log10(l1min), np.log10(l1max), int(l1max-l1min+1))
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')
        int_ll = integrate.simps(int_1, x=phi1, even='avg')
        result = 1./int_ll

        result *= L**2

        if not np.isfinite(result):
            result = 0.

        if result < 0.:
            print(L)

        return result

    def calc_var(self, est):
        data = np.zeros((self.Nl, 4))
        data[:, 0] = np.copy(self.L)
        pool = Pool(ncpus=4)
        n_est = len(est)
        for i_est in range(n_est):
            XY = est[i_est]
            print("Computing variance for " + XY)

            def f(l):
                return self.var_individual(l, XY)

            data[:, i_est+1] = np.array(pool.map(f, self.L))
        np.savetxt(self.var_out, data)

    def interp_var(self, est):

        print("Loading variances")

        self.var_d = {}
        data = np.genfromtxt(self.var_out)
        L = data[:, 0]

        n_est = len(est)
        for i_est in range(n_est):
            XY = est[i_est]
            var = data[:, i_est+1].copy()
            self.var_d[XY] = interp1d(L, var, kind='linear',
                                      bounds_error=False, fill_value=0.)

    def plot_var(self, est):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        data = np.genfromtxt("input/CAMB/Julien_lenspotentialCls.dat")
        L = data[:, 0]
        ax.plot(L, data[:, 5], 'r-', lw=1.5, label=r'signal')

        n_est = len(est)
        for i_est in range(n_est):
            XY = est[i_est]
            ax.plot(self.L, self.L*(self.L+1)*self.var_d[XY](self.L)/(2*np.pi), c=plt.cm.rainbow(i_est/6.), lw=1.5, label=XY)
            # ax.plot(self.L, self.var_d[XY](self.L), c=plt.cm.rainbow(i_est/6.), lw=1.5, label=XY)

        ax.legend(loc=2, fontsize='8')
        ax.set_xscale('log')
        ax.set_yscale('log', nonposy='mask')
        ax.set_xlabel(r'$L$', fontsize=16)
        ax.set_ylabel(r'$L(L+1)C_L^{dd}/2\pi$', fontsize=16)
        ax.set_ylim((3.e-11, 0.1))
        ax.set_xlim((2., 4.e4))
        plt.show()

    def trispec_covariance(self, L, XY):
        """
        Trispectrum contribution to the Covariance of the QE for XY
        """
        l1min = self.l1Min
        l1max = self.l1Max

        if L > 2*l1max:  # L = l1 + l2 thus max L = 2*l1
            return 0.

        def integrand(l_1, phil):

            l_2 = self.l2(L, l_1, phil)

            """
            if l_1 < l1min or l_2 < l1min or l_1 > l1max or l_2 > l1max:
                return 0.
            # """

            result = self.F_XY(L, l_1, phil, XY)
            result *= 2*l_1  # **2
            # d^2l_1 = dl_1*l_1*dphi1
            """factor of 2 above because phi integral is symmetric. Thus we've
            put instead of 0 to 2pi, 2 times 0 to pi
            Also, l_1^2 instead of l_1 if we are taking log spacing for
            l_1"""
            result /= (2.*np.pi)**2

            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            result[idx] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            return result

        # """
        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')
        int_l1 = integrate.simps(int_1, x=phi1, even='avg')
        result = int_l1**2
        # """

        result *= self.var_d[XY](L)**2
        if XY == 'XX':
            result *= self.im.trispecXX(L)
        elif XY == 'YY':
            result *= self.im.trispecYY(L)
        else:
            result *= self.im.trispecXY(L)

        result *= 1./L**2

        if not np.isfinite(result):
            result = 0.
        return result

    def calc_trispec(self, est):
        data = np.zeros((self.Nl, 1+len(est)))
        data[:, 0] = np.copy(self.L)
        pool = Pool(ncpus=4)

        tri = {}
        n_est = len(est)
        counter = 1
        for i_est in range(n_est):
            XY = est[i_est]

            print("Computing trispectrum for " + XY + "-" + XY)

            def ft(l):
                return self.trispec_covariance(l, XY)

            tri[XY+XY] = np.array(pool.map(ft, self.L))
            data[:, counter] = tri[XY+XY]
            counter += 1
            # self.tri_d = {}
            # self.tri_d[XY+XY] = interp1d(self.L, tri[XY+XY], kind='linear', bounds_error=False, fill_value=0.)
        np.savetxt(self.trispec_out, data)

    def interp_trispec(self, est):
        print("Interpolating trispectra")

        self.tri_d = {}
        data = np.genfromtxt(self.trispec_out)
        L = data[:, 0]

        n_est = len(est)
        counter = 1
        for i_est in range(n_est):
            XY = est[i_est]
            norm = data[:, counter].copy()
            self.tri_d[XY+XY] = interp1d(L, norm, kind='linear', bounds_error=False, fill_value=0.)
            counter += 1

    def prim_bispec(self, L, XY):
        """
        Primary brispectrum contribution to the Covariance of the QE for XX
        """
        l1min = self.l1Min
        l1max = self.l1Max

        if L > 2*l1max:  # L = l1 + l2 thus max L = 2*l1
            return 0.

        def integrand(l_1, phil):

            l_2 = self.l2(L, l_1, phil)

            """
            if l_1 < l1min or l_2 < l1min or l_1 > l1max or l_2 > l1max:
                return 0.
            # """

            result = self.F_XY(L, l_1, phil, XY)
            if XY == 'XX':
                result *= self.im.bispecXX(L)  # (np.array([[L, l_1, l_2]]))
            elif XY == 'YY':
                result *= self.im.bispecYY(L)  # ([L, l_1, l_2])
            else:
                result *= self.im.bispecXY(L)  # ([L, l_1, l_2])

            result *= 2*l_1  # **2
            # d^2l_1 = dl_1*l_1*dphi1
            """factor of 2 above because phi integral is symmetric. Thus we've
            put instead of 0 to 2pi, 2 times 0 to pi
            Also, l_1^2 instead of l_1 if we are taking log spacing for
            l_1"""
            result /= (2.*np.pi)**2

            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            result[idx] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            return result

        # """
        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        # print l1min, l1max
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')
        int_l1 = integrate.simps(int_1, x=phi1, even='avg')
        result = int_l1
        # """
        #
        result *= 2*self.var_d[XY](L)  # **2
        result *= 1./L  # **2

        # # bispec here should be from B^{dgg} instead of B^{kappa gg}
        # as the code calculates the bias terms for C_l{dd}
        result *= -2./L

        if not np.isfinite(result):
            result = 0.
        return result

    def calc_primbispec(self, est):
        data = np.zeros((self.Nl, 1+len(est)))
        data[:, 0] = np.copy(self.L)
        pool = Pool(ncpus=4)

        primbi = {}
        n_est = len(est)
        counter = 1
        for i_est in range(n_est):
            XY = est[i_est]

            print("Computing primary bispectrum for " + XY + "-" + XY)

            def ft(l):
                return self.prim_bispec(l, XY)

            primbi[XY+XY] = np.array(pool.map(ft, self.L))
            data[:, counter] = primbi[XY+XY]
            counter += 1
            # self.pbi_d = {}
            # self.pbi_d[XY+XY] = interp1d(self.L, primbi[XY+XY], kind='linear', bounds_error=False, fill_value=0.)
            # if XY == 'XX' or XY == 'YY':
            #     primbi[XY+'CMB'] = self.L*primbi[XY+XY]/2./self.var_d[XY](self.L)
        np.savetxt(self.primbispec_out, data)

    def interp_primbispec(self, est):
        print("Interpolating primary bispectra")

        self.pbi_d = {}
        data = np.genfromtxt(self.primbispec_out)
        L = data[:, 0]

        n_est = len(est)
        counter = 1
        for i_est in range(n_est):
            XY = est[i_est]
            norm = data[:, counter].copy()
            self.pbi_d[XY+XY] = interp1d(L, norm, kind='linear', bounds_error=False, fill_value=0.)
            if XY == 'XX' or XY == 'YY':
                res = self.pbi_d[XY+XY](L)/2.  # /self.var_d[XY](L)
                self.pbi_d[XY+'CMB'] = interp1d(L, res, kind='linear', bounds_error=False, fill_value=0.)
            counter += 1

    def sec_bispec(self, L, XY):
        """
        Secondary brispectrum contribution to the Covariance of the QE for XX
        """
        l1min = self.l1Min
        l1max = self.l1Max

        if L > 2*l1max:  # L = l1 + l2 thus max L = 2*l1
            return 0.

        def integrand(l_1, phi_1):

            l_2 = self.l2(L, l_1, phi_1)
            phi_2 = self.phi2(L, l_1, phi_1)

            def integrand2(l_3, phi_3):
                l_4 = self.l4(L, l_3, phi_3)
                phi_4 = self.phi4(L, l_3, phi_3)
                phi_13 = phi_1 - phi_3
                phi_14 = phi_1 - phi_4
                phi_23 = phi_2 - phi_3
                phi_24 = phi_2 - phi_4

                l_13 = self.mag_l(l_1, l_3, phi_13)
                l_14 = self.mag_l(l_1, l_4, phi_14)
                l_23 = self.mag_l(l_2, l_3, phi_23)
                l_24 = self.mag_l(l_2, l_4, phi_24)

                # # bispec here should be from B^{phi gg} instead of
                # B^{kappa gg} which I get from Manu, thus multiplying by
                # 2/L**2
                if XY == 'XX':  #  or XY == 'YY':
                    # result = self.f_XY_bispec(l_3, l_4, phi_3-phi_4, XY)  # self.f_XY_bispec(l_3, l_4, phi_3-phi_4, XY)
                    result = self.f_XY_bispec(l_1, l_3, phi_13, XY)*self.im.bispecXX(L)*(-2./l_13**2)  # ([l_13, l_2, l_4])  # l__1, l__2, phi_12, XY
                    result += self.f_XY_bispec(l_1, l_4, phi_14, XY)*self.im.bispecXX(L)*(-2./l_14**2)  # ([l_14, l_2, l_3])
                    result += self.f_XY_bispec(l_2, l_3, phi_23, XY)*self.im.bispecXX(L)*(-2./l_23**2)  # ([l_23, l_1, l_4])
                    result += self.f_XY_bispec(l_2, l_4, phi_24, XY)*self.im.bispecXX(L)*(-2./l_24**2)  # ([l_24, l_1, l_3])
                    result[~np.isfinite(result)] = 0.
                elif XY == 'YY':
                    result = self.f_XY_bispec(l_1, l_3, phi_13, XY)*self.im.bispecYY(L)*(-2./l_13**2)  # ([l_13, l_2, l_4])  # l__1, l__2, phi_12, XY
                    result += self.f_XY_bispec(l_1, l_4, phi_14, XY)*self.im.bispecYY(L)*(-2./l_14**2)  # ([l_14, l_2, l_3])
                    result += self.f_XY_bispec(l_2, l_3, phi_23, XY)*self.im.bispecYY(L)*(-2./l_23**2)  # ([l_23, l_1, l_4])
                    result += self.f_XY_bispec(l_2, l_4, phi_24, XY)*self.im.bispecYY(L)*(-2./l_24**2)  # ([l_24, l_1, l_3])
                    result[~np.isfinite(result)] = 0.
                else:
                    result = self.f_XY_bispec(l_1, l_3, phi_13, 'XX')*self.im.bispecYY(L)*(-2./l_13**2)  # ([l_13, l_2, l_4])
                    result += self.f_XY_bispec(l_2, l_4, phi_24, 'YY')*self.im.bispecXX(L)*(-2./l_24**2)  # ([l_24, l_1, l_3])
                    result[~np.isfinite(result)] = 0.

                result *= self.F_XY_bispec(L, l_3, phi_3, XY)  # *2
                result *= l_3  # **2
                # d^2l_1 = dl_1*l_1*dphi1
                """factor of 2 above because phi integral is symmetric. Thus we've
                put instead of 0 to 2pi, 2 times 0 to pi
                Also, l_1^2 instead of l_1 if we are taking log spacing for
                l_1"""
                result /= (2.*np.pi)**2

                idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max) | (l_3 < l1min) | (l_3 > l1max) | (l_4 < l1min) | (l_4 > l1max))
                result[idx] = 0.
                """
                for nr in range(len(result)):
                    if not np.isfinite(result[nr]):
                        result[nr] = 0.
                """
                return result

            l3 = np.linspace(l1min, l1max, int(l1max-l1min+1))
            phi3 = np.linspace(0., 2*np.pi, self.N_phi)
            int_3 = np.zeros(len(phi3))
            for i in range(len(phi3)):
                intgnd = integrand2(l3, phi3[i])
                int_3[i] = integrate.simps(intgnd, x=l3, even='avg')
            int_l3 = integrate.simps(int_3, x=phi3, even='avg')
            result = int_l3
            # result *= self.var_d[XY](L)/L**2

            """
            if l_1 < l1min or l_2 < l1min or l_1 > l1max or l_2 > l1max:
                return 0.
            # """

            result *= self.F_XY(L, l_1, phi_1, XY)
            result *= 2*l_1  # **2
            # d^2l_1 = dl_1*l_1*dphi1
            """factor of 2 above because phi integral is symmetric. Thus we've
            put instead of 0 to 2pi, 2 times 0 to pi
            Also, l_1^2 instead of l_1 if we are taking log spacing for
            l_1"""
            result /= (2.*np.pi)**2

            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            result[idx] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            return result

        # """
        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')
        int_l1 = integrate.simps(int_1, x=phi1, even='avg')
        result = int_l1
        # """

        result *= self.var_d[XY](L)**2
        """
        if XY == 'XX':
            result *= self.im.bispecXX(L)
        elif XY == 'YY':
            result *= self.im.bispecYY(L)
        else:
            result *= 1.  # self.im.bispecXY(L) (bispectrum factors already multiplied at the integration level)
        # """
        result *= 1./L**2

        if not np.isfinite(result):
            result = 0.
        return result

    def calc_secbispec(self, est):
        data = np.zeros((self.Nl, 1+len(est)))
        data[:, 0] = np.copy(self.L)
        pool = Pool(ncpus=4)

        secbi = {}
        n_est = len(est)
        counter = 1
        for i_est in range(n_est):
            XY = est[i_est]

            print("Computing secondary bispectrum for " + XY + "-" + XY)

            def ft(l):
                return self.sec_bispec(l, XY)

            secbi[XY+XY] = np.array(pool.map(ft, self.L))
            data[:, counter] = secbi[XY+XY]
            counter += 1
            # self.pbi_d = {}
            # self.pbi_d[XY+XY] = interp1d(self.L, primbi[XY+XY], kind='linear', bounds_error=False, fill_value=0.)
        np.savetxt(self.secbispec_out, data)

    def interp_secbispec(self, est):
        print("Interpolating secondary bispectra")

        self.sbi_d = {}
        data = np.genfromtxt(self.secbispec_out)
        L = data[:, 0]

        n_est = len(est)
        counter = 1
        for i_est in range(n_est):
            XY = est[i_est]
            norm = data[:, counter].copy()
            self.sbi_d[XY+XY] = interp1d(L, norm, kind='linear', bounds_error=False, fill_value=0.)
            counter += 1

    def covariance_G(self, L, XY, AB):
        """
        Covariance of the QE for XY and AB map choice. 
        """
        l1min = self.l1Min
        l1max = self.l1Max

        if L > 2*l1max:  # L = l1 + l2 thus max L = 2*l1
            return 0.

        def integrand(l_1, phil):

            l_2 = self.l2(L, l_1, phil)
            phi2 = self.phi2(L, l_1, phil)

            """
            if l_1 < l1min or l_2 < l1min or l_1 > l1max or l_2 > l1max:
                return 0.
            # """

            if XY == 'XX' and AB == 'YY':
                a = self.F_XY(L, l_1, phil, AB)*self.im.totalXY(l_1)*self.im.totalXY(l_2)
                a += self.F_XY(L, l_2, phi2, AB)*self.im.totalXY(l_1)*self.im.totalXY(l_2)
                # print a

            elif XY == 'XX' and AB == 'XY':
                a = self.F_XY(L, l_1, phil, AB)*self.im.totalXX(l_1)*self.im.totalXY(l_2)
                a += self.F_XY(L, l_2, phi2, AB)*self.im.totalXY(l_1)*self.im.totalXX(l_2)

            elif XY == 'YY' and AB == 'XY':
                a = self.F_XY(L, l_1, phil, AB)*self.im.totalXY(l_1)*self.im.totalYY(l_2)
                a += self.F_XY(L, l_2, phi2, AB)*self.im.totalYY(l_1)*self.im.totalXY(l_2)

            result = a*self.F_XY(L, l_1, phil, XY)
            result *= 2*l_1  # **2
            # d^2l_1 = dl_1*l_1*dphi1
            """factor of 2 above because phi integral is symmetric. Thus we've
            put instead of 0 to 2pi, 2 times 0 to pi
            Also, l_1^2 instead of l_1 if we are taking log spacing for
            l_1"""
            result /= (2.*np.pi)**2

            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            result[idx] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            return result

        # """
        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')
        int_l1 = integrate.simps(int_1, x=phi1, even='avg')
        result = int_l1
        # """

        result *= self.var_d[XY](L)*self.var_d[AB](L)
        result *= 1./L**2

        if not np.isfinite(result):
            result = 0.
        return result

    def covariance_nG(self, L, XY, AB):
        """
        Covariance of the QE for XY and AB map choice. 
        """
        l1min = self.l1Min
        l1max = self.l1Max

        if L > 2*l1max:  # L = l1 + l2 thus max L = 2*l1
            return 0.

        def integrand(l_1, phi_1):

            l_2 = self.l2(L, l_1, phi_1)
            phi_2 = self.phi2(L, l_1, phi_1)

            if XY == 'XX' and AB == 'YY':
                a = self.F_XY(L, l_1, phi_1, XY)*self.var_d[XY](L)*self.im.bispecXX(L)  # ([L, l_1, l_2])
                a += self.F_XY(L, l_1, phi_1, AB)*self.var_d[AB](L)*self.im.bispecYY(L)  # ([L, l_1, l_2])
                # a *= self.var_d[XY](L)*self.var_d[AB](L)
                a *= 1./L
                a *= -2./L  # B^{d g g} to B^{kappa g g}

            elif (XY == 'XX')*(AB == 'XY') or (XY == 'YY')*(AB == 'XY'):
                if XY == 'XX' and AB == 'XY':  # XX-XY pair also has a primary bispectrum component
                    a1 = self.F_XY(L, l_1, phi_1, XY)*self.var_d[XY](L)*self.im.bispecXX(L)
                    a1 *= 1./L
                    a1 *= -2./L  # B^{d g g} to B^{kappa g g}
                elif XY == 'YY' and AB == 'XY':
                    a1 = self.F_XY(L, l_1, phi_1, XY)*self.var_d[XY](L)*self.im.bispecYY(L)
                    a1 *= 1./L
                    a1 *= -2./L  # B^{d g g} to B^{kappa g g}

                def integrand2(l_3, phi_3):
                    l_4 = self.l4(L, l_3, phi_3)
                    phi_4 = self.phi4(L, l_3, phi_3)
                    phi_13 = phi_1 - phi_3
                    phi_14 = phi_1 - phi_4
                    phi_23 = phi_2 - phi_3
                    phi_24 = phi_2 - phi_4

                    l_13 = self.mag_l(l_1, l_3, phi_13)
                    l_14 = self.mag_l(l_1, l_4, phi_14)
                    l_23 = self.mag_l(l_2, l_3, phi_23)
                    l_24 = self.mag_l(l_2, l_4, phi_24)

                    if XY == 'XX' and AB == 'XY':
                        result = self.f_XY_bispec(l_1, l_4, phi_14, AB)*self.im.bispecXX(L)*(-2./l_14**2)  # ([l_14, l_2, l_3])
                        result += self.f_XY_bispec(l_2, l_4, phi_24, AB)*self.im.bispecXX(L)*(-2./l_24**2)  # ([l_24, l_1, l_3])
                        result[~np.isfinite(result)] = 0.
                        # result *= self.im.bispecXX(L)
                    elif XY == 'YY' and AB == 'XY':
                        result = self.f_XY_bispec(l_3, l_1, -phi_13, AB)*self.im.bispecYY(L)*(-2./l_13**2)  # ([l_13, l_2, l_4])
                        result += self.f_XY_bispec(l_3, l_2, -phi_23, AB)*self.im.bispecYY(L)*(-2./l_23**2)  # ([l_23, l_1, l_4])
                        result[~np.isfinite(result)] = 0.
                        # result *= self.im.bispecYY(L)

                    result *= self.F_XY_bispec(L, l_3, phi_3, AB)  # *2
                    result *= l_3  # **2
                    # d^2l_1 = dl_1*l_1*dphi1
                    """factor of 2 above because phi integral is symmetric. Thus we've
                    put instead of 0 to 2pi, 2 times 0 to pi
                    Also, l_1^2 instead of l_1 if we are taking log spacing for
                    l_1"""
                    result /= (2.*np.pi)**2
                    idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max) | (l_3 < l1min) | (l_3 > l1max) | (l_4 < l1min) | (l_4 > l1max))
                    result[idx] = 0.
                    return result

                l3 = np.linspace(l1min, l1max, int(l1max-l1min+1))
                phi3 = np.linspace(0., 2*np.pi, self.N_phi)
                int_3 = np.zeros(len(phi3))
                for i in range(len(phi3)):
                    intgnd = integrand2(l3, phi3[i])
                    int_3[i] = integrate.simps(intgnd, x=l3, even='avg')
                int_l3 = integrate.simps(int_3, x=phi3, even='avg')
                a = int_l3
                a *= self.F_XY(L, l_1, phi_1, XY)
                a *= self.var_d[XY](L)*self.var_d[AB](L)
                a *= 1./L**2
                # a[np.isnan(a)] = 0.

            if (XY == 'XX')*(AB == 'XY') or (XY == 'YY')*(AB == 'XY'):
                result = a + a1  # +   # *self.F_XY(L, l_1, phi_1, XY)
            else:
                result = a  # *self.F_XY(L, l_1, phi_1, XY)

            result *= 2*l_1  # **2
            # d^2l_1 = dl_1*l_1*dphi1
            """factor of 2 above because phi integral is symmetric. Thus we've
            put instead of 0 to 2pi, 2 times 0 to pi
            Also, l_1^2 instead of l_1 if we are taking log spacing for
            l_1"""
            result /= (2.*np.pi)**2

            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            result[idx] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            return result

        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')
        int_l1 = integrate.simps(int_1, x=phi1, even='avg')
        result = int_l1

        # result *= self.var_d[XY](L)*self.var_d[AB](L)
        # result *= 1./L**2

        if not np.isfinite(result):
            result = 0.
        return result

    def calc_cov(self, est):
        data = np.zeros((self.Nl, 5))
        data[:, 0] = np.copy(self.L)
        data_ng = np.zeros((self.Nl, 7))
        data_ng[:, 0] = np.copy(self.L)

        pool = Pool(ncpus=4)

        cov_XY_AB_G = {}
        cov_XY_AB_nG = {}
        # tri = {}
        n_est = len(est)
        counter = 1
        counter_ng = 0
        for i_est in range(n_est):
            XY = est[i_est]
            cov_XY_AB_G[XY+XY] = self.var_d[XY](self.L)
            cov_XY_AB_nG[XY+XY] = self.tri_d[XY+XY](self.L) + self.pbi_d[XY+XY](self.L) + self.sbi_d[XY+XY](self.L)

            counter_ng += 1
            data_ng[:, counter_ng] = cov_XY_AB_nG[XY+XY]
            for i2 in range(i_est+1, n_est):
                AB = est[i2]
                print("Computing covariance for " + XY + "-" + AB)

                def f_G(l):
                    return self.covariance_G(l, XY, AB)

                def f_nG(l):
                    return self.covariance_nG(l, XY, AB)

                cov_XY_AB_G[XY+AB] = np.array(pool.map(f_G, self.L))
                cov_XY_AB_nG[XY+AB] = np.array(pool.map(f_nG, self.L))

                """
                cov = np.zeros(len(self.L))
                for nL in range(len(self.L)):
                    cov[nL] = f(self.L[nL])
                    print XY, AB, self.L[nL]  # , cov[nL]
                cov_XY_AB[XY+AB] = cov
                """
                data[:, counter] = cov_XY_AB_G[XY+AB] + cov_XY_AB_nG[XY+AB]
                counter += 1
                counter_ng += 1
                data_ng[:, counter_ng] = cov_XY_AB_nG[XY+AB]
                # counter_ng += 1

        """
        # min variance estimator noise
        n_mv = np.zeros(self.Nl)
        for el in range(self.Nl):
            covmat = np.zeros((n_est, n_est))
            for i_est in range(n_est):
                XY = est[i_est]
                covmat[i_est, i_est] = self.var_d[XY](self.L[el]) + self.tri_d[XY+XY](self.L[el]) + self.pbi_d[XY+XY](self.L[el]) + self.sbi_d[XY+XY](self.L[el])
                for i2 in range(i_est+1, n_est):
                    AB = est[i2]
                    covmat[i_est, i2] = covmat[i2, i_est] = cov_XY_AB_G[XY+AB][el]+cov_XY_AB_nG[XY+AB][el]
            # invert the matrix
            try:
                invcov = np.linalg.inv(covmat)
                n_mv[el] = 1./np.sum(invcov)
                # if el == 6:
                #     print (self.L[el])
                #     np.savetxt('output/covmat.txt', covmat)
            except:
                print("exception while inverting the covariance matrix at L = %s !" % str(el))
                pass

        data[:, -1] = n_mv
        """
        np.savetxt(self.covar_out, data)
        np.savetxt(self.covar_out_ng, data_ng)

    def interp_cov(self, est):
        print("Interpolating covariances")

        self.cov_d = {}
        data = np.genfromtxt(self.covar_out)
        L = data[:, 0]

        self.cov_d_ng = {}
        data_ng = np.genfromtxt(self.covar_out_ng)

        n_est = len(est)
        counter = 1
        counter_ng = 0

        for i_est in range(n_est):
            XY = est[i_est]
            counter_ng += 1
            norm_ng = data_ng[:, counter_ng].copy()
            self.cov_d_ng[XY+XY] = interp1d(L, norm_ng, kind='linear', bounds_error=False, fill_value=0.)
            for i2 in range(i_est+1, n_est):
                AB = est[i2]
                norm = data[:, counter].copy()
                self.cov_d[XY+AB] = interp1d(L, norm, kind='linear', bounds_error=False, fill_value=0.)
                counter += 1

                counter_ng += 1
                norm_ng = data_ng[:, counter_ng].copy()
                self.cov_d_ng[XY+AB] = interp1d(L, norm_ng, kind='linear', bounds_error=False, fill_value=0.)

        nmv = data[:, -1]
        self.var_d['mv'] = interp1d(L, nmv, kind='linear', bounds_error=False, fill_value=0.)

    def plot_cov(self, est):
        lines = ["-", "--", "-."]
        cl = ["b", "r", "g", "k"]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        data2 = np.genfromtxt("input/CAMB/Julien_lenspotentialCls.dat")
        L = data2[:, 0]
        ax.plot(L, data2[:, 5], 'r-', lw=1.5, label=r'signal')

        n_est = len(est)
        for i_est in range(n_est):
            XY = est[i_est]
            # if XY == 'XX' or XY == 'YY' :
            varind = self.var_d[XY](self.L)
            tri = self.tri_d[XY+XY](self.L)
            pbi = self.pbi_d[XY+XY](self.L)
            sbi = self.sbi_d[XY+XY](self.L)
            tot = varind + tri + pbi + sbi

            ax.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=plt.cm.rainbow(i_est/3.), ls = '-', lw=1.5, label=XY+' total')
            # ax.plot(self.L, self.L*(self.L+1)*varind/(2*np.pi), c=plt.cm.rainbow(i_est+1/3.), ls = '--', lw=1.0, label=XY+' auto')
            # ax.plot(self.L, self.L*(self.L+1)*tri/(2*np.pi), c=plt.cm.rainbow(i_est+2/3.), ls='-.', lw=1.0, label=XY+' tri')
            # ax.plot(self.L, self.L*(self.L+1)*pbi/(2*np.pi), c='r', ls=':', lw=1.0, label=XY+' p-bi')
            # ax.plot(self.L, self.L*(self.L+1)*sbi/(2*np.pi), c='b', ls=':', lw=1.0, label=XY+' s-bi')
            """
            else:
                varind = self.var_d[XY](self.L)
                tot = varind
                ax.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=plt.cm.rainbow(i_est/3.), ls = '-', lw=1.5, label=XY+' total')
            """ 

        # ax.plot(self.L, self.L*(self.L+1)*tri/(2*np.pi), 'p--', lw=1.5, label='XX Trispectrum')
        # totXX = self.var_d['XX'](self.L)  + self.tri_d['XX'+'XX'](self.L)
        # ax.plot(self.L, self.L*(self.L+1)*totXX/(2*np.pi), 'p-.', lw=1.5, label='XX Total')

        ax.plot(self.L, self.L*(self.L+1)*self.var_d['mv'](self.L)/(2*np.pi), 'k--', lw=1.5, label='min var')

        ax.legend(fontsize='12', loc='lower left')  # , labelspacing=0.1)
        ax.set_xscale('log')
        ax.set_yscale('log', nonposy='mask')
        ax.set_xlabel(r'$L$', fontsize=16)
        ax.set_ylabel(r'$L(L+1)C_L^{dd}/2\pi$', fontsize=16)
        # ax.set_ylim((1e-9, 4.e-6))
        ax.set_xlim((2., 4.e3))
        ax.tick_params(axis='both', labelsize=12)
        plt.show()

    def plot_cov_XX(self, est):
        lines = ["-", "--", "-.", ":"]
        cl = ["b", "r", "g", "k"]

        fig = plt.figure(figsize=(11.5, 7))
        ax = fig.add_subplot(111)

        # data2 = np.genfromtxt("input/CAMB/Julien_lenspotentialCls.dat")
        data2 = np.genfromtxt('input/p2d_limz5lenshalofit.txt')
        L = data2[:, 0]
        # ax.plot(L, data2[:, 5], 'r-', lw=1.5, label=r'signal')
        ax.plot(L, data2[:, 1]+data2[:, 2], 'k-', lw=2.5, label=r'Lensing signal')

        dl = self.L*(self.L+1)/4  # (2*np.pi)
        n_est = len(est)
        for i_est in range(n_est):
            XY = est[i_est]
            varind = self.var_d[XY](self.L)
            tri = self.tri_d[XY+XY](self.L)
            pbi = self.pbi_d[XY+XY](self.L)
            sbi = self.sbi_d[XY+XY](self.L)
            tot = varind + tri + pbi   # + sbi

            # ax.plot(self.L, dl*tot, c=cl[0], ls=lines[0], lw=1.5, label='Total auto')
            ax.plot(self.L, dl*varind, c=cl[0], ls=lines[0], lw=2.0, label=r'Noise bias $N^{(0)}$')
            ax.plot(self.L, dl*np.abs(pbi), c=cl[0], ls=lines[2], lw=2.0, label='Primary bispectrum interloper bias')
            # ax.plot(self.L, dl*np.abs(sbi), c=cl[0], ls=lines[3], lw=1.0, label='Secondary bispectrum interloper bias')
            ax.plot(self.L, dl*tri, c=cl[0], ls=lines[1], lw=2.0, label='Trispectrum interloper bias')

        # ax.plot(self.L, self.L*(self.L+1)*tri/(2*np.pi), 'p--', lw=1.5, label='XX Trispectrum')
        # totXX = self.var_d['XX'](self.L)  + self.tri_d['XX'+'XX'](self.L)
        # ax.plot(self.L, self.L*(self.L+1)*totXX/(2*np.pi), 'p-.', lw=1.5, label='XX Total')

        # ax.plot(self.L, self.L*(self.L+1)*self.var_d['mv'](self.L)/(2*np.pi), 'k', lw=1.5, label='min var')

        ax.legend(fontsize='18', bbox_to_anchor=(0.38, 0.45))  # , labelspacing=0.1)
        ax.set_xscale('log')
        ax.set_yscale('log', nonposy='mask')
        ax.set_xlabel(r'$L$', fontsize=24)
        ax.set_ylabel(r'$C_L^{\hat{\kappa}_{XX}}$', fontsize=26)
        ax.set_ylim((1e-10))  # , 4.e-6))
        ax.set_xlim((5., 2.e3))
        ax.tick_params(axis='both', labelsize=20)
        plt.show()
        # plt.savefig('output/Figures/HO02_curves_onlyLya_allcomp_Ha_0.11_addedto_Lya_5_1halotrispec.pdf', bbox_inches="tight")

    def plot_cov_gaussian_total(self, est):
        lines = ["-", "--", "-."]
        cl = ["g", "b", "r", "k"]

        fig = plt.figure(figsize=(10.5, 7))
        ax = fig.add_subplot(111)

        # data2 = np.genfromtxt("input/CAMB/Julien_lenspotentialCls.dat")
        data2 = np.genfromtxt('input/p2d_limz5lenshalofit.txt')
        L = data2[:, 0]
        ax.plot(L, data2[:, 1]+data2[:, 2], 'k-', lw=1.5, label=r'Lensing signal')

        dl = self.L*(self.L+1)/4  # (2*np.pi)
        n_est = len(est)
        for i_est in range(n_est):
            XY = est[i_est]
            varind = self.var_d[XY](self.L)
            tri = self.tri_d[XY+XY](self.L)
            pbi = self.pbi_d[XY+XY](self.L)
            sbi = self.sbi_d[XY+XY](self.L)
            tot = varind + tri + pbi + sbi

            ax.plot(self.L, dl*tot, c=cl[i_est], ls = '-', lw=1.5, label='noise '+XY+' total')
            ax.plot(self.L, dl*varind, c=cl[i_est], ls = '--', lw=1.0, label='noise '+XY+' auto')

        ax.legend(fontsize='12', loc='upper right')  # , frameon=False)  # , labelspacing=0.1)
        ax.set_xscale('log')
        ax.set_yscale('log', nonposy='mask')
        ax.set_xlabel(r'$L$', fontsize=16)
        ax.set_ylabel(r'$C_L^{\kappa \kappa}$', fontsize=16)
        ax.set_ylim((1e-9))  # , 4.e-6))
        ax.set_xlim((5., 4.e3))
        ax.tick_params(axis='both', labelsize=12)
        plt.show()

    def plot_kappa_pairs(self, pairs):
        lines = ["-", "--", "-."]
        cl = ["g", "b", "r", "c", "tab:gray", "m"]

        fig = plt.figure(figsize=(11.0, 8.3))
        ax = fig.add_subplot(111)
        # ax = plt.subplot(111)

        # data2 = np.genfromtxt("input/CAMB/Julien_lenspotentialCls.dat")
        data2 = np.genfromtxt('input/p2d_limz5lenshalofit.txt')
        L = data2[:, 0]
        ax.plot(L, data2[:, 1]+data2[:, 2], 'k-', lw=2.5, label=r'Lensing signal')

        dl = self.L*(self.L+1)/4  # (2*np.pi)
        n_p = len(pairs)
        coeff = 1  # 1e9  # 1e23
        for i_p in range(n_p):
            if pairs[i_p] == 'XX-XX':
                XY = 'XX'
                varind = self.var_d[XY](self.L)
                tri = self.tri_d[XY+XY](self.L)
                pbi = self.pbi_d[XY+XY](self.L)
                sbi = self.sbi_d[XY+XY](self.L)
                tot = np.abs(tri + pbi)  #  + sbi)   # + varind
                ax.plot(self.L, coeff*dl*tot, c=cl[i_p], ls='-', lw=2.0, label=XY+'-'+XY)
                print ('XX-XX', dl[:5], tot[:5])
            elif pairs[i_p] == 'XX-YY':
                tot = np.abs(self.cov_d_ng['XXYY'](self.L))
                ax.plot(self.L, coeff*dl*tot, c=cl[i_p], ls='-', lw=2.0, label='XX-YY')
                print ('XX-YY', dl[:5], tot[:5])
            elif pairs[i_p] == 'XX-XY':
                tot = np.abs(self.cov_d_ng['XXXY'](self.L))
                ax.plot(self.L, coeff*dl*tot, c=cl[i_p], ls='-', lw=2.0, label='XX-XY')
                print ('XX-XY', dl[:5], tot[:5])
            elif pairs[i_p] == 'XY-XY':
                tot = np.abs(self.cov_d_ng['XYXY'](self.L))
                ax.plot(self.L, coeff*dl*tot, c=cl[i_p], ls='-', lw=2.0, label='XY-XY')
                print ('XY-XY', dl[:5], tot[:5])
            elif pairs[i_p] == 'XX-CMB':
                tot = np.abs(self.pbi_d['XXCMB'](self.L))
                ax.plot(self.L, coeff*dl*tot, c=cl[i_p], ls='-', lw=2.0, label='XX-CMB')
                print ('XX-CMB', tot[:5])
            """
            elif pairs[i_p] == 'XY-CMB':
                tot = np.zeros(len(self.L))
                ax.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label='XY-CMB')
            elif pairs[i_p] == 'XY-YZ':
                tot = np.zeros(len(self.L))
                ax.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label='XY-YZ')
            elif pairs[i_p] == 'XY-ZW':
                tot = np.zeros(len(self.L))
                ax.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label='XY-ZW')
            """

        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_ylim((1e-10, 0.02))
        # ax.set_ylim((2., 0.09e23))
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_ylabel(r'${\rm Foreground \: bias \: to} \: C_L^{\hat{\kappa}_{XX}}$', fontsize=22)
        # plt.title(r'${\rm Foreground \: bias \: to} \: C_L^{\kappa \kappa}$', fontsize=16)

        divider = make_axes_locatable(ax)
        axLin = divider.append_axes("bottom", size=2.0, pad=0.02, sharex=ax)
        for i_p in range(n_p):
            if pairs[i_p] == 'XY-CMB':
                tot = np.zeros(len(self.L))
                axLin.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=2.0, label='XY-CMB')
            elif pairs[i_p] == 'XY-YZ':
                tot = np.zeros(len(self.L))
                axLin.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label='XY-YZ')
            elif pairs[i_p] == 'XY-ZW':
                tot = np.zeros(len(self.L))
                axLin.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label='XY-ZW')
        # axLin.plot(xdomain, np.sin(xdomain))
        axLin.set_xscale('log')
        axLin.set_ylim((-1e-14, 1e-14))
        # """
        y_ticks = [-1e-14, 0., 1e-14]  # , 1e-10]
        # ytick_labels = [r'-0.5 $\times 10^{-10}$', '0.0', r'0.5 $\times 10^{-10}$']  # , r'10^{-10}$']
        ytick_labels = [r'$-10^{-14}$', '0.0', r'$10^{-14}$']  # , r'10^{-10}$']
        # """
        axLin.set_yticks(y_ticks)
        axLin.set_yticklabels(ytick_labels)
        """
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        # ax.yaxis.set_ticks_position('left')
        # ax.spines['left'].set_position(('data',0))
        plt.xticks(x_ticks, xtick_labels, fontsize='small', rotation=45)
        """
        # axLin.set_ylim((-0.5, 2.))
        axLin.spines['top'].set_visible(False)
        axLin.set_xlabel(r'$L$', fontsize=24)
        # ax.set_ylabel(r'$L(L+1)N_L^{\phi \phi}/2\pi$', fontsize=16)
        ax.set_xlim((5., 2.e3))
        axLin.set_xlim((5., 2.e3))
        ax.tick_params(axis='both', labelsize=20)
        axLin.tick_params(labelsize=20)
        ax.legend(fontsize='16', loc='lower right', bbox_to_anchor=(0.95, -0.06), frameon=False)  # , labelspacing=0.1)
        axLin.legend(fontsize='16', loc='upper right', bbox_to_anchor=(0.87, 1.02), frameon=False)  # , labelspacing=0.1)
        plt.show()
        # plt.savefig('output/Figures/biasto_kappaxkappa.pdf', bbox_inches="tight")

    def plot_cov_pair(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.L, self.L*(self.L+1)*np.abs(self.cov_d['XXYY'](self.L))/(2*np.pi), label='XX-YY')
        ax.plot(self.L, self.L*(self.L+1)*np.abs(self.cov_d['XXXY'](self.L))/(2*np.pi), label='XX-XY')
        ax.plot(self.L, self.L*(self.L+1)*np.abs(self.cov_d['YYXY'](self.L))/(2*np.pi), label='YY-XY')
        ax.legend(loc=2, fontsize='12')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$L$', fontsize=16)
        ax.set_ylabel(r'$L(L+1)|{N_L}^\mathrm{cov}|/2\pi$', fontsize=16)
        # ax.set_ylim((1.e-15))
        ax.set_xlim((2., 3.e4))
        ax.tick_params(axis='both', labelsize=12)
        plt.show()

    def corr_coef(self, est):
        self.pear_coeff = {}
        n_est = len(est)
        for i_est in range(n_est):
            XY = est[i_est]
            for i2 in range(i_est+1, n_est):
                AB = est[i2]
                if self.cov_d[XY+AB] == 0:
                    break
                else:
                    # np.seterr(divide='raise', invalid='raise')
                    # print XY, AB, self.N_d[XY](self.L), self.N_d[AB](self.L)
                    numerator = self.cov_d[XY+AB](self.L)
                    denominator = np.sqrt(self.var_d[XY](self.L)*self.var_d[AB](self.L))
                    # result[:, counter] = numerator/denominator
                    self.pear_coeff[XY+AB] = numerator/denominator
                    # print counter, XY+AB
        return self.pear_coeff

    def plot_corrcoef(self, est):
        pear = self.corr_coef(est)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.L, np.abs(pear['XXYY']), label='XX-YY')
        ax.plot(self.L, np.abs(pear['XXXY']), label='XX-XY')
        ax.plot(self.L, np.abs(pear['YYXY']), label='YY-XY')
        # for i in range(4):
        #     ax.plot(self.L, pear[:, i], label=pairs[i])
        ax.legend(prop={'size': 14}, frameon=False)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xlabel(r'$L$', fontsize=16)
        ax.set_ylabel(r'$|{N_{AB}}(L)| / \sqrt{{N_{AA}(L)} \times {N_{BB}(L)}}$', fontsize=16)
        ax.set_ylim(ymin=-0.001)
        ax.set_xlim((2., self.l1Max))
        ax.set_xscale('log')
        ax.tick_params(axis='both', labelsize=14)
        plt.show()

    def snr_clkappakappa(self, est, fsky):
        nl = 51
        ell = np.logspace(np.log10(2.), np.log10(2*self.l1Max+1.), nl, 10.)
        data2 = np.genfromtxt("input/p2d_cmblenshalofit.txt")
        L = data2[:, 0]
        cl_kappacmb = interp1d(L, data2[:, 1]+data2[:, 2],
                               kind='linear', bounds_error=False,
                               fill_value=0.)
        kappalim = np.loadtxt('input/p2d_limz5lenshalofit.txt')
        cl_kappaXY = interp1d(kappalim[:, 0], kappalim[:, 1]+kappalim[:, 2],
                              kind='linear', bounds_error=False, fill_value=0.)

        kappacmblim = np.loadtxt('input/p2d_cmblenslimz5lenshalofit.txt')
        cl_crosskappa = interp1d(kappacmblim[:, 0], kappacmblim[:, 1]+kappacmblim[:, 2],
                                 kind='linear', bounds_error=False,
                                 fill_value=0.)
        # ###### adding files for cl_kappa_null ###############
        kappanull = np.loadtxt('input/p2d_nulllenshalofit.txt')
        cl_kappanull = interp1d(kappanull[:, 0],
                                kappanull[:, 1]+kappanull[:, 2], kind='linear',
                                bounds_error=False, fill_value=0.)

        kappacmbnull = np.loadtxt('input/p2d_cmblensnulllenshalofit.txt')
        cl_crosskappanull = interp1d(kappacmbnull[:, 0],
                                     kappacmbnull[:, 1]+kappacmbnull[:, 2],
                                     kind='linear', bounds_error=False,
                                     fill_value=0.)

        alpha = self.im.alphanull
        """
        for noise on kappa_null, we ignore the sec. bispectrum noise caused by
        kappa_XY. So we only consider N0 noise. K_null is combination of
        kappa_null = kappa_CMB + alpha*kappa_XY1 - (1+alpha)*kappa_XY2
        Here N0 from different kappa are uncorrelated. That is why, when we
        take the auto-power spectrum of kappa_null, apart from the auto- and
        cross-CMB lensing power spectra, we get
        N0tot = alpha^2 * N0_kappaXY1 + (1+alpha)^2 * N0_kappaXY2 + N0_kappacmb
        Here we neglect N0_kapacmb assuming it is signal dominated.
        Another assumption we make is that N0_kappaXY1 = _kappaXY2 in our case
        as we have XY1 at z=5 and XY2 at z=6. So our final variance comes out
        to be
        (2.*alpha^2 + 2.*alpha + 1) * N0_kappaXY
        """
        alphaterm = 2.*alpha**2 + 2.*alpha + 1  # (alpha^2 + (1+alpha)^2)

        # n_est = len(est)
        # for i_est in range(n_est):
        XY = 'XY'  # est[i_est]
        """
        varind = self.var_d[XY](self.L)
        tri = self.tri_d[XY+XY](self.L)
        pbi = self.pbi_d[XY+XY](self.L)
        sbi = self.sbi_d[XY+XY](self.L)
        cl_XY_totnoise = varind + tri + pbi + sbi
        cl_phiXY_tot = cl_phiXY(self.L) + cl_XY_totnoise
        """
        lcen = np.zeros(nl-1)
        fcrossbin = np.zeros(nl-1)
        ftotnoisebin = np.zeros(nl-1)
        fkappacmbbin = np.zeros(nl-1)
        fkappaXYbin = np.zeros(nl-1)
        fkappaXYtot_bin = np.zeros(nl-1)
        fgaussnoisebin = np.zeros(nl-1)
        fkappaXYgauss_bin = np.zeros(nl-1)

        fkappanullbin = np.zeros(nl-1)
        fcrossnullkappabin = np.zeros(nl-1)
        ftotnoisenullbin = np.zeros(nl-1)
        fkappanulltot_bin = np.zeros(nl-1)

        for i in range(nl-1):
            l1 = ell[i]
            l2 = ell[i+1]
            lcen[i] = (l1+l2)/2.
            el = np.linspace(l1, l2, int(l2-l1+1))
            dl = el*(el+1.)/(4.)
            deltal = l2-l1
            """
            if i==1:
                print (el)
                print (deltal)
                print (cl_crosskappa(el))
            """
            fcrossbin[i] = np.sum(cl_crosskappa(el))/deltal
            fkappacmbbin[i] = np.sum(cl_kappacmb(el))/deltal
            fkappaXYbin[i] = np.sum(cl_kappaXY(el))/deltal
            ftotnoisebin[i] = np.sum(dl*(self.var_d[XY](el)+self.tri_d[XY+XY](el)+self.pbi_d[XY+XY](el)))/deltal  # +self.sbi_d[XY+XY](el)
            fgaussnoisebin[i] = np.sum(dl*self.var_d[XY](el))/deltal
            fkappaXYtot_bin[i] = fkappaXYbin[i]+ftotnoisebin[i]
            fkappaXYgauss_bin[i] = fkappaXYbin[i]+fgaussnoisebin[i]

            # bins for null kappa calculation
            fcrossnullkappabin[i] = np.sum(cl_crosskappanull(el))/deltal
            fkappanullbin[i] = np.sum(cl_kappanull(el))/deltal
            ftotnoisenullbin[i] = np.sum(dl*(alphaterm*self.var_d[XY](el)))/deltal  # +self.sbi_d[XY+XY](el)
            fkappanulltot_bin[i] = fkappanullbin[i]+ftotnoisenullbin[i]

        # print (fcrossbin[:5])
        num = (2*lcen+1)*deltal*fsky*fcrossbin**2
        # print (np.shape(cross), np.shape(cibtot), np.shape(galtot))
        denom = fcrossbin**2 + fkappacmbbin*fkappaXYtot_bin
        denom_g = fcrossbin**2 + fkappacmbbin*fkappaXYgauss_bin
        # print (denom[:5])
        # print (num[:5])
        snr2 = np.cumsum(num/denom)
        snr_g2 = np.cumsum(num/denom_g)
        # print (snr2[:5])
        snrbin = np.sqrt(snr2)
        snrbin_g = np.sqrt(snr_g2)
        print (snrbin[-1])
        print (snrbin_g[-1])

        # SNR calcultion for null
        numnull = (2*lcen+1)*deltal*fsky*fcrossnullkappabin**2
        # print (np.shape(cross), np.shape(cibtot), np.shape(galtot))
        denomnull = fcrossnullkappabin**2 + fkappacmbbin*fkappanulltot_bin
        # print (denom[:5])
        # print (num[:5])
        snrnull2 = np.cumsum(numnull/denomnull)
        # print (snr2[:5])
        snrbinnull = np.sqrt(snrnull2)
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(lcen, snrbin, 'k-', lw=1.5)  # , label=r'CUMSNR')
        # ax.legend(fontsize='12', loc='upper left')  # , labelspacing=0.1)
        ax.set_xscale('log')
        # ax.set_yscale('log', nonposy='mask')
        ax.set_xlabel(r'$L$', fontsize=16)
        ax.set_ylabel(r'SNR', fontsize=16)
        # ax.set_ylim((1e-9, 4.e-6))
        ax.set_xlim((5., 4.e3))
        ax.tick_params(axis='both', labelsize=12)
        plt.show()
        # """
        """
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(lcen, fcrossbin, 'k-', lw=1.5, label=r'signal')
        ax2.plot(lcen, ftotnoisebin, 'b-', lw=1.5, label=r'noise')
        ax2.legend(fontsize='12', loc='upper right')  # , labelspacing=0.1)
        ax2.set_xscale('log')
        ax2.set_yscale('log', nonposy='mask')
        ax2.set_xlabel(r'$L$', fontsize=16)
        ax2.set_ylabel(r'$C_L^{\kappa_{\rm CMB} \kappa_{\rm XY}}$', fontsize=16)
        # ax.set_ylim((1e-9, 4.e-6))
        ax2.set_xlim((5., 4.e3))
        ax2.tick_params(axis='both', labelsize=12)
        plt.show()
        # """
        return lcen, snrbin, snrbin_g, snrbinnull

    def snr_clkappanullkappacmb(self, est, fsky):
        nl = 51
        ell = np.logspace(np.log10(2.), np.log10(2*self.l1Max+1.), nl, 10.)
        data2 = np.genfromtxt("input/p2d_cmblenshalofit.txt")
        L = data2[:, 0]
        cl_kappacmb = interp1d(L, data2[:, 1]+data2[:, 2],
                               kind='linear', bounds_error=False,
                               fill_value=0.)

        kappanull = np.loadtxt('input/p2d_nulllenshalofit.txt')
        cl_kappanull = interp1d(kappanull[:, 0], kappanull[:, 1]+kappanull[:, 2],
                                kind='linear', bounds_error=False, fill_value=0.)

        kappacmbnull = np.loadtxt('input/p2d_cmblensnulllenshalofit.txt')
        cl_crosskappa = interp1d(kappacmbnull[:, 0], kappacmbnull[:, 1]+kappacmbnull[:, 2],
                                 kind='linear', bounds_error=False,
                                 fill_value=0.)
        
        alpha = self.im.alphanull
        alphaterm = 2.*alpha**2 + 2.*alpha + 1  # (alpha^2 + (1+alpha)^2)
        # n_est = len(est)
        # for i_est in range(n_est):
        XY = 'XY'  # est[i_est]
        """
        varind = self.var_d[XY](self.L)
        tri = self.tri_d[XY+XY](self.L)
        pbi = self.pbi_d[XY+XY](self.L)
        sbi = self.sbi_d[XY+XY](self.L)
        cl_XY_totnoise = varind + tri + pbi + sbi
        cl_phiXY_tot = cl_phiXY(self.L) + cl_XY_totnoise
        """
        lcen = np.zeros(nl-1)
        fcrossbin = np.zeros(nl-1)
        ftotnoisebin = np.zeros(nl-1)
        fkappacmbbin = np.zeros(nl-1)
        fkappanullbin = np.zeros(nl-1)
        fkappanulltot_bin = np.zeros(nl-1)
        fgaussnoisebin = np.zeros(nl-1)
        fkappanullgauss_bin = np.zeros(nl-1)

        for i in range(nl-1):
            l1 = ell[i]
            l2 = ell[i+1]
            lcen[i] = (l1+l2)/2.
            el = np.linspace(l1, l2, int(l2-l1+1))
            dl = el*(el+1.)/(4.)
            deltal = l2-l1

            fcrossbin[i] = np.sum(cl_crosskappa(el))/deltal
            fkappacmbbin[i] = np.sum(cl_kappacmb(el))/deltal
            fkappanullbin[i] = np.sum(cl_kappanull(el))/deltal
            ftotnoisebin[i] = np.sum(dl*(alphaterm*self.var_d[XY](el)))/deltal  # +self.sbi_d[XY+XY](el)
            fgaussnoisebin[i] = np.sum(dl*alphaterm*self.var_d[XY](el))/deltal
            fkappanulltot_bin[i] = fkappanullbin[i]+ftotnoisebin[i]
            fkappanullgauss_bin[i] = fkappanullbin[i]+fgaussnoisebin[i]

        num = (2*lcen+1)*deltal*fsky*fcrossbin**2
        denom = fcrossbin**2 + fkappacmbbin*fkappanulltot_bin
        denom_g = fcrossbin**2 + fkappacmbbin*fkappanullgauss_bin
        snr2 = np.cumsum(num/denom)
        snr_g2 = np.cumsum(num/denom_g)
        snrbin = np.sqrt(snr2)
        snrbin_g = np.sqrt(snr_g2)
        print (snrbin[-1])
        print (snrbin_g[-1])
        return lcen, snrbin, snrbin_g

    def A_bias_XX(self, fsky):
        kappalim = np.loadtxt('input/p2d_limz5lenshalofit.txt')
        cl_kappaXY = interp1d(kappalim[:, 0], kappalim[:, 1]+kappalim[:, 2],
                              kind='linear', bounds_error=False, fill_value=0.)
        
        # n_est = len(est)
        # for i_est in range(n_est):
        XY = 'XX'  # est[i_est]
        lcen = np.zeros(self.Nl-1)
        ftotnoisebin = np.zeros(self.Nl-1)
        fkappaXYbin = np.zeros(self.Nl-1)
        fkappaXYtot_bin = np.zeros(self.Nl-1)
        fgaussnoisebin = np.zeros(self.Nl-1)
        fkappaXYgauss_bin = np.zeros(self.Nl-1)
        fbias_bin = np.zeros(self.Nl-1)
        # fbias_Ahat_bin = np.zeros(self.Nl-1)

        for i in range(self.Nl-1):
            l1 = self.L[i]
            l2 = self.L[i+1]
            lcen[i] = (l1+l2)/2.
            el = np.linspace(l1, l2, int(l2-l1+1))
            dl = el*(el+1.)/(4.)
            deltal = l2-l1
            """
            if i==1:
                print (el)
                print (deltal)
                print (cl_crosskappa(el))
            """
            fkappaXYbin[i] = np.sum(cl_kappaXY(el))/deltal
            ftotnoisebin[i] = np.sum(dl*(self.var_d[XY](el)+self.tri_d[XY+XY](el)+self.pbi_d[XY+XY](el)))/deltal  # +self.sbi_d[XY+XY](el)
            fgaussnoisebin[i] = np.sum(dl*self.var_d[XY](el))/deltal
            fkappaXYtot_bin[i] = fkappaXYbin[i]+ftotnoisebin[i]
            fkappaXYgauss_bin[i] = fkappaXYbin[i]+fgaussnoisebin[i]
            fbias_bin[i] = np.sum(dl*(self.tri_d[XY+XY](el)+np.abs(self.pbi_d[XY+XY](el))))/deltal  # +self.sbi_d[XY+XY](el)

        sigma_denom = (2*lcen+1)*deltal*fsky
        sigma_num = 2*fkappaXYtot_bin**2
        sigma_bin2 = sigma_num/sigma_denom

        num = np.cumsum(fbias_bin*fkappaXYbin/sigma_bin2)
        denom = np.cumsum(fkappaXYbin**2/sigma_bin2)

        bias_Ahat = num/denom
        return lcen, bias_Ahat


if __name__ == '__main__':
    import time
    # import imp
    # import cell_cmb
    # imp.reload(cell_cmb)
    from cell_im import *

    time0 = time()

    fam = "serif"
    plt.rcParams["font.family"] = fam

    # """
    # first
    first = {"name": "Lya_CII", "lMin": 30., "lMax": 1500.,
          "zc": 5.}

    im = Cell_im(first)

    l_est = lensing_estimator(im)

    # est = ['XX']  # , 'YY', 'XY']
    est = ['XX', 'YY', 'XY']

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
            l_est = lensing_estimator(im)
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
            l_est = lensing_estimator(im)
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
