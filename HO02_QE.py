from imports import *


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
        self.L = np.logspace(np.log10(1.), np.log10(2*self.l1Max+1.), 51, 10.)
        # self.L = np.linspace(1., 201., 1001)
        self.Nl = len(self.L)
        self.N_phi = 50  # number of steps for angular integration steps
        # reduce to 50 if you need around 0.6% max accuracy till L = 3000
        # from 200 to 400, there is just 0.03% change in the noise curves till L=3000
        self.var_out = 'output/HO02_variance_individual_%s_lmin%s_lmax%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax))
        self.covar_out = 'output/HO02_covariance_%s_lmin%s_lmax%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax))
        self.primbispec_out = 'output/HO02_primbispec_%s_lmin%s_lmax%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax))
        self.secbispec_out = 'output/HO02_secbispec_%s_lmin%s_lmax%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax))
        self.trispec_out = 'output/HO02_trispec_%s_lmin%s_lmax%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax))
        self.covar_out_ng = 'output/HO02_covariance_nonGaussian_%s_lmin%s_lmax%s.txt' % (self.name, str(self.im.lMin), str(self.im.lMax))

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
            # print result
            # sys.exit()
        elif XY == 'YY':
            result = self.im.unlensedYY(l_1)*Ldotl_1
            result += self.im.unlensedYY(l_2)*Ldotl_2
        elif XY == 'XY':
            result = self.im.unlensedXY(l_1)*Ldotl_1
            result += self.im.unlensedXY(l_2)*Ldotl_2
        """
        if XY == 'TT':
            result = self.cmb.lensedTT(l_1)*Ldotl_1
            result += self.cmb.lensedTT(l_2)*Ldotl_2
            # print result
            # sys.exit()
        # """
        # result *= 2. / L**2

        return result

    def f_XY_bispec(self, l__1, l__2, phi_12, XY):
        """
        lensing response such that
        <X_l1 Y_{L-l1}> = f_XY(l1, L-l1)*\phi_L.
        Here this is defined for calculating the terms in the secondary
        bispectrum. Here this is defined as
        f_XY(l_1, l_2) = C_(l_1)*[l_1**2 + l_1*l_2*cos(phi12)] + C_(l_2)*[l_2**2 + l_1*l_2*cos(phi12)]
        """

        L12dotl__1 = l__1**2 + l__1*l__2*np.cos(phi_12)
        L12dotl__2 = l__2**2 + l__1*l__2*np.cos(phi_12)
        # """
        if XY == 'XX':
            result = self.im.unlensedXX(l__1)*L12dotl__1
            result += self.im.unlensedXX(l__2)*L12dotl__2
            # print result
            # sys.exit()
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
            """
            for i in range(len(denominator)):
                if denominator[i] == 0.:
                    print L, l_1[i], self.im.totalXX(l_1)[i], l_2[i], self.im.totalXX(l_2)[i], denominator[i]
            # """
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
        X and Y.
        """

        l_2 = self.l4(L, l__1, phi_1)
        # phi2 = self.phi4(L, l__1, phi_1)
        phi12 = self.phi34(L, l__1, phi_1)  # phi2 - phi_1
        # print (L, l__1, l_2, l__1+l_2)

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
            # idx = np.where((l_2 < l1min) | (l_2 > l1max))
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            # print idx
            result[idx] = 0.
            # idx2 = np.where(~np.isfinite(result))
            # print idx2
            # result[idx2] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            return result


        # unlensed = np.loadtxt('input/cl_%s_%s.txt' % (self.name, self.im.zc))
        # l1 = unlensed[:, 0]
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

            # idx = np.where((l_2 < l1min) | (l_2 > l1max))
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            # print idx, l_1[idx], l_2[idx]
            result[idx] = 0.
            # idx2 = np.where(~np.isfinite(result))
            # result[idx2] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            # print result
            return result

        # """
        # unlensed = np.loadtxt('input/cl_%s_%s.txt' % (self.name, self.im.zc))
        # l1 = unlensed[:, 0]
        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        # print l1min, l1max
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')
        int_l1 = integrate.simps(int_1, x=phi1, even='avg')
        result = int_l1**2
        # print result
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
            result *= 2*l_1  # **2
            # d^2l_1 = dl_1*l_1*dphi1
            """factor of 2 above because phi integral is symmetric. Thus we've
            put instead of 0 to 2pi, 2 times 0 to pi
            Also, l_1^2 instead of l_1 if we are taking log spacing for
            l_1"""
            result /= (2.*np.pi)**2

            # idx = np.where((l_2 < l1min) | (l_2 > l1max))
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            # print idx, l_1[idx], l_2[idx]
            result[idx] = 0.
            # idx2 = np.where(~np.isfinite(result))
            # result[idx2] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            # print result
            return result

        # """
        # unlensed = np.loadtxt('input/cl_%s_%s.txt' % (self.name, self.im.zc))
        # l1 = unlensed[:, 0]
        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        # print l1min, l1max
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')
        int_l1 = integrate.simps(int_1, x=phi1, even='avg')
        result = int_l1
        # print result
        # """

        result *= 2*self.var_d[XY](L)**2
        if XY == 'XX':
            result *= self.im.bispecXX(L)
        elif XY == 'YY':
            result *= self.im.bispecYY(L)
        else:
            result *= self.im.bispecXY(L)

        result *= 1./L**2

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
                res = L*self.pbi_d[XY+XY](L)/2./self.var_d[XY](L)
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

                if XY == 'XX' or XY == 'YY':
                    # result = self.f_XY_bispec(l_3, l_4, phi_3-phi_4, XY)  # self.f_XY_bispec(l_3, l_4, phi_3-phi_4, XY)
                    result = self.f_XY_bispec(l_1, l_3, phi_13, XY)  # l__1, l__2, phi_12, XY
                    result += self.f_XY_bispec(l_1, l_4, phi_14, XY)
                    result += self.f_XY_bispec(l_2, l_3, phi_23, XY)
                    result += self.f_XY_bispec(l_2, l_4, phi_24, XY)
                else:
                    result = self.im.bispecYY(L)*self.f_XY_bispec(l_1, l_3, phi_13, 'XX')
                    result += self.im.bispecXX(L)*self.f_XY_bispec(l_2, l_4, phi_13, 'YY')

                result *= self.F_XY_bispec(L, l_3, phi_3, XY)  # *2
                result *= l_3  # **2
                # d^2l_1 = dl_1*l_1*dphi1
                """factor of 2 above because phi integral is symmetric. Thus we've
                put instead of 0 to 2pi, 2 times 0 to pi
                Also, l_1^2 instead of l_1 if we are taking log spacing for
                l_1"""
                result /= (2.*np.pi)**2

                # idx = np.where((l_2 < l1min) | (l_2 > l1max))
                idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max) | (l_3 < l1min) | (l_3 > l1max) | (l_4 < l1min) | (l_4 > l1max))
                # print idx, l_1[idx], l_2[idx]
                result[idx] = 0.
                # idx2 = np.where(~np.isfinite(result))
                # result[idx2] = 0.
                """
                for nr in range(len(result)):
                    if not np.isfinite(result[nr]):
                        result[nr] = 0.
                """
                # print result
                return result

            l3 = np.linspace(l1min, l1max, int(l1max-l1min+1))
            # print l1min, l1max
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

            # idx = np.where((l_2 < l1min) | (l_2 > l1max))
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            # print idx, l_1[idx], l_2[idx]
            result[idx] = 0.
            # idx2 = np.where(~np.isfinite(result))
            # result[idx2] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            # print result
            return result

        # """
        # unlensed = np.loadtxt('input/cl_%s_%s.txt' % (self.name, self.im.zc))
        # l1 = unlensed[:, 0]
        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        # print l1min, l1max
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')
        int_l1 = integrate.simps(int_1, x=phi1, even='avg')
        result = int_l1
        # print result
        # """

        result *= self.var_d[XY](L)**2
        if XY == 'XX':
            result *= self.im.bispecXX(L)
        elif XY == 'YY':
            result *= self.im.bispecYY(L)
        else:
            result *= 1.  # self.im.bispecXY(L) (bispectrum factors already multiplied at the integration level)

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

            # idx = np.where((l_2 < l1min) | (l_2 > l1max))
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            # print idx, l_1[idx], l_2[idx]
            result[idx] = 0.
            # idx2 = np.where(~np.isfinite(result))
            # result[idx2] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            # print result
            return result

        # """
        # unlensed = np.loadtxt('input/cl_%s_%s.txt' % (self.name, self.im.zc))
        # l1 = unlensed[:, 0]
        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        # print l1min, l1max
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')
        int_l1 = integrate.simps(int_1, x=phi1, even='avg')
        result = int_l1
        # print result
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
                a = self.F_XY(L, l_1, phi_1, XY)*self.im.bispecXX(L)
                a += self.F_XY(L, l_1, phi_1, AB)*self.im.bispecYY(L)

            elif (XY == 'XX')*(AB == 'XY') or (XY == 'YY')*(AB == 'XY'):

                def integrand2(l_3, phi_3):
                    l_4 = self.l4(L, l_3, phi_3)
                    phi_4 = self.phi4(L, l_3, phi_3)
                    phi_13 = phi_1 - phi_3
                    phi_14 = phi_1 - phi_4
                    phi_23 = phi_2 - phi_3
                    phi_24 = phi_2 - phi_4

                    if XY == 'XX' and AB == 'XY':
                        result = self.f_XY_bispec(l_1, l_4, phi_14, AB)  # l__1, l__2, phi_12, XY
                        result += self.f_XY_bispec(l_2, l_4, phi_24, AB)
                        result *= self.im.bispecXX(L)
                    elif XY == 'YY' and AB == 'XY':
                        result = self.f_XY_bispec(l_3, l_1, -phi_13, AB)
                        result += self.f_XY_bispec(l_3, l_2, -phi_23, AB)
                        result *= self.im.bispecYY(L)

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
                # print l1min, l1max
                phi3 = np.linspace(0., 2*np.pi, self.N_phi)
                int_3 = np.zeros(len(phi3))
                for i in range(len(phi3)):
                    intgnd = integrand2(l3, phi3[i])
                    int_3[i] = integrate.simps(intgnd, x=l3, even='avg')
                int_l3 = integrate.simps(int_3, x=phi3, even='avg')
                a = int_l3

                a *= self.F_XY(L, l_1, phi_1, XY)

            result = a  # *self.F_XY(L, l_1, phi_1, XY)
            result *= 2*l_1  # **2
            # d^2l_1 = dl_1*l_1*dphi1
            """factor of 2 above because phi integral is symmetric. Thus we've
            put instead of 0 to 2pi, 2 times 0 to pi
            Also, l_1^2 instead of l_1 if we are taking log spacing for
            l_1"""
            result /= (2.*np.pi)**2

            # idx = np.where((l_2 < l1min) | (l_2 > l1max))
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            # print idx, l_1[idx], l_2[idx]
            result[idx] = 0.
            # idx2 = np.where(~np.isfinite(result))
            # result[idx2] = 0.
            """
            for nr in range(len(result)):
                if not np.isfinite(result[nr]):
                    result[nr] = 0.
            """
            # print result
            return result

        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')
        int_l1 = integrate.simps(int_1, x=phi1, even='avg')
        result = int_l1

        result *= self.var_d[XY](L)*self.var_d[AB](L)
        result *= 1./L**2

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
                # np.savetxt('covmat.txt', covmat)
            except:
                print("exception while inverting the covariance matrix at L = %s !" % str(el))
                pass

        data[:, -1] = n_mv
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
            if XY == 'XX' or XY == 'YY':
                varind = self.var_d[XY](self.L)
                tri = self.tri_d[XY+XY](self.L)
                pbi = self.pbi_d[XY+XY](self.L)
                sbi = self.sbi_d[XY+XY](self.L)
                tot = varind + tri + pbi + sbi

                # ax.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=plt.cm.rainbow(i_est/3.), ls = '-', lw=1.5, label=XY+' total')
                # ax.plot(self.L, self.L*(self.L+1)*varind/(2*np.pi), c=plt.cm.rainbow(i_est+1/3.), ls = '--', lw=1.0, label=XY+' auto')
                # ax.plot(self.L, self.L*(self.L+1)*tri/(2*np.pi), c=plt.cm.rainbow(i_est+2/3.), ls='-.', lw=1.0, label=XY+' tri')
                ax.plot(self.L, self.L*(self.L+1)*pbi/(2*np.pi), c='r', ls=':', lw=1.0, label=XY+' p-bi')
                ax.plot(self.L, self.L*(self.L+1)*sbi/(2*np.pi), c='b', ls=':', lw=1.0, label=XY+' s-bi')
            else:
                varind = self.var_d[XY](self.L)
                tot = varind
                ax.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=plt.cm.rainbow(i_est/3.), ls = '-', lw=1.5, label=XY+' total')
                

        # ax.plot(self.L, self.L*(self.L+1)*tri/(2*np.pi), 'p--', lw=1.5, label='XX Trispectrum')
        # totXX = self.var_d['XX'](self.L)  + self.tri_d['XX'+'XX'](self.L)
        # ax.plot(self.L, self.L*(self.L+1)*totXX/(2*np.pi), 'p-.', lw=1.5, label='XX Total')

        # ax.plot(self.L, self.L*(self.L+1)*self.var_d['mv'](self.L)/(2*np.pi), 'k', lw=1.5, label='min var')

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

        fig = plt.figure()
        ax = fig.add_subplot(111)

        data2 = np.genfromtxt("input/CAMB/Julien_lenspotentialCls.dat")
        L = data2[:, 0]
        ax.plot(L, data2[:, 5], 'r-', lw=1.5, label=r'signal')

        n_est = len(est)
        for i_est in range(n_est):
            XY = est[i_est]
            varind = self.var_d[XY](self.L)
            tri = self.tri_d[XY+XY](self.L)
            pbi = self.pbi_d[XY+XY](self.L)
            sbi = self.sbi_d[XY+XY](self.L)
            tot = varind + tri + pbi + sbi

            ax.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=cl[0], ls = lines[0], lw=1.5, label=XY+' total')
            ax.plot(self.L, self.L*(self.L+1)*varind/(2*np.pi), c=cl[0], ls = lines[0], lw=1.0, alpha=0.5, label=XY+' auto')
            ax.plot(self.L, self.L*(self.L+1)*tri/(2*np.pi), c=cl[0], ls= lines[1], lw=1.0, label=XY+' tri')
            ax.plot(self.L, self.L*(self.L+1)*pbi/(2*np.pi), c=cl[0], ls= lines[2], lw=1.0, label=XY+' p-bi')
            ax.plot(self.L, self.L*(self.L+1)*sbi/(2*np.pi), c=cl[0], ls= lines[3], lw=1.0, label=XY+' s-bi')

        # ax.plot(self.L, self.L*(self.L+1)*tri/(2*np.pi), 'p--', lw=1.5, label='XX Trispectrum')
        # totXX = self.var_d['XX'](self.L)  + self.tri_d['XX'+'XX'](self.L)
        # ax.plot(self.L, self.L*(self.L+1)*totXX/(2*np.pi), 'p-.', lw=1.5, label='XX Total')

        # ax.plot(self.L, self.L*(self.L+1)*self.var_d['mv'](self.L)/(2*np.pi), 'k', lw=1.5, label='min var')

        ax.legend(fontsize='12', loc='lower left')  # , labelspacing=0.1)
        ax.set_xscale('log')
        ax.set_yscale('log', nonposy='mask')
        ax.set_xlabel(r'$L$', fontsize=16)
        ax.set_ylabel(r'$L(L+1)N_L^{\phi \phi}/2\pi$', fontsize=16)
        # ax.set_ylim((1e-9, 4.e-6))
        ax.set_xlim((5., 4.e3))
        ax.tick_params(axis='both', labelsize=12)
        plt.show()

    def plot_cov_gaussian_total(self, est):
        lines = ["-", "--", "-."]
        cl = ["g", "b", "r", "k"]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        data2 = np.genfromtxt("input/CAMB/Julien_lenspotentialCls.dat")
        L = data2[:, 0]
        ax.plot(L, data2[:, 5], 'k-', lw=1.5, label=r'signal')

        n_est = len(est)
        for i_est in range(n_est):
            XY = est[i_est]
            varind = self.var_d[XY](self.L)
            tri = self.tri_d[XY+XY](self.L)
            pbi = self.pbi_d[XY+XY](self.L)
            sbi = self.sbi_d[XY+XY](self.L)
            tot = varind + tri + pbi + sbi

            ax.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_est], ls = '-', lw=1.5, label=XY+' total')
            ax.plot(self.L, self.L*(self.L+1)*varind/(2*np.pi), c=cl[i_est], ls = '--', lw=1.0, label=XY+' auto')

        ax.legend(fontsize='12', loc='lower left')  # , labelspacing=0.1)
        ax.set_xscale('log')
        ax.set_yscale('log', nonposy='mask')
        ax.set_xlabel(r'$L$', fontsize=16)
        ax.set_ylabel(r'$L(L+1)N_L^{\phi \phi}/2\pi$', fontsize=16)
        # ax.set_ylim((1e-9, 4.e-6))
        ax.set_xlim((5., 4.e3))
        ax.tick_params(axis='both', labelsize=12)
        plt.show()

    def plot_kappa_pairs(self, pairs):
        lines = ["-", "--", "-."]
        cl = ["g", "b", "r", "k", "m"]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax = plt.subplot(111)

        data2 = np.genfromtxt("input/CAMB/Julien_lenspotentialCls.dat")
        L = data2[:, 0]

        n_p = len(pairs)
        coeff = 1e23
        for i_p in range(n_p):
            if pairs[i_p] == 'XX-XX':
                XY = 'XX'
                varind = self.var_d[XY](self.L)
                tri = self.tri_d[XY+XY](self.L)
                pbi = self.pbi_d[XY+XY](self.L)
                sbi = self.sbi_d[XY+XY](self.L)
                tot = tri + pbi + sbi + varind
                ax.plot(self.L, coeff*self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label=XY+'-'+XY)
            elif pairs[i_p] == 'XX-YY':
                tot = self.cov_d_ng['XXYY'](self.L)
                ax.plot(self.L, coeff*self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label='XX-YY')
            elif pairs[i_p] == 'XX-XY':
                tot = self.cov_d_ng['XXXY'](self.L)
                ax.plot(self.L, coeff*self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label='XX-XY')
            elif pairs[i_p] == 'XX-CMB':
                tot = self.pbi_d['XXCMB'](self.L)
                ax.plot(self.L, coeff*self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label='XX-CMB')
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
        # ax.set_ylim((3e-23, 0.09))
        ax.set_ylim((2., 0.09e23))
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        plt.title(r'bias to $L(L+1)C_L^{\phi \phi}/2\pi \: \times 10^{23}$', fontsize=16)

        divider = make_axes_locatable(ax)
        axLin = divider.append_axes("bottom", size=2.0, pad=0.02, sharex=ax)
        for i_p in range(n_p):
            if pairs[i_p] == 'XY-CMB':
                tot = np.zeros(len(self.L))
                axLin.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label='XY-CMB')
            elif pairs[i_p] == 'XY-YZ':
                tot = np.zeros(len(self.L))
                axLin.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label='XY-YZ')
            elif pairs[i_p] == 'XY-ZW':
                tot = np.zeros(len(self.L))
                axLin.plot(self.L, self.L*(self.L+1)*tot/(2*np.pi), c=cl[i_p], ls = '-', lw=1.5, label='XY-ZW')
        # axLin.plot(xdomain, np.sin(xdomain))
        axLin.set_xscale('linear')
        # axLin.set_ylim((-0.1, 3e-23))
        axLin.set_ylim((-0.5, 2.))
        axLin.spines['top'].set_visible(False)
        axLin.set_xlabel(r'$L$', fontsize=18)
        # ax.set_ylabel(r'$L(L+1)N_L^{\phi \phi}/2\pi$', fontsize=16)
        ax.set_xlim((5., 4.e3))
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(fontsize='14', loc='lower right')  # , labelspacing=0.1)
        axLin.legend(fontsize='14', loc='upper right')  # , labelspacing=0.1)
        plt.show()

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


if __name__ == '__main__':
    import time
    # import imp
    # import cell_cmb
    # imp.reload(cell_cmb)
    from cell_im import *

    time0 = time()
    
    # """
    # first
    first = {"name": "Lya_CII", "lMin": 30., "lMax": 4000.,
          "zc": 5.}

    im = Cell_im(first)

    l_est = lensing_estimator(im)

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
    # l_est.plot_cov_XX(['XY'])
    # l_est.plot_cov_pair()
    # l_est.plot_cov_gaussian_total(est)
    # l_est.plot_corrcoef(est)
    pairs = ['XX-XX', 'XX-YY', 'XX-XY', 'XX-CMB', 'XY-CMB']
    l_est.plot_kappa_pairs(pairs)
    # """
    print(time()-time0)
