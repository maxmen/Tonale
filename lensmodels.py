from astropy import constants as const
from astropy.constants import c
from astropy import units as u
import astropy.units as units
import numpy as np
from skimage import measure
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib import cm

class genlen(object):
    def __init__(self):
        self.initialized=True

    def setGrid(self,theta=None, thetax=None, thetay=None, compute_potential=False):
        """
        set Grid covering the source plane, and compute lens maps such as convergence,
        shear, and deflection angles
        :param theta: a numpy array setting the pixel positions along one axis
        :param compute_potential: if True, the map of the potential is computed. This is time and memory consuming,
        therefore it is False by default.
        It is neccessary to set this flag to True if you want to compute time delay surfaces.

        :return: the function does not return any value, but the maps will be accessible as
        properties:
        example:
        kappa=genlens.ka
        g1=genlens.g1
        g2=genlens.g2
        a1=genlens.a1
        a2=genlens.a2
        pot=genlens.potential
        """

        # construct a mesh:
        if thetax is None or thetay is None:
            self.theta1, self.theta2 = np.meshgrid(theta, theta)
            self.thetax = theta
            self.thetay = theta
        else:
            self.theta1, self.theta2 = np.meshgrid(thetax, thetay)
            if np.round(thetax[1]-thetax[0], 6) == np.round(thetay[1]-thetay[0], 6):
                self.thetax = thetax
                self.thetay = thetay
            else:
                raise Exception('thetax and thetay must have the same pixel scale '
                                + str(thetax[1]-thetax[0])
                                + ', ' + str(thetay[1]-thetay[0]))



        self.g1, self.g2 = self.gamma(self.theta1, self.theta2)
        self.ka = self.kappa(self.theta1, self.theta2)
        self.a1,self.a2=self.angle(self.theta1, self.theta2)
        self.pot = self.potential(self.theta1, self.theta2)

        self.size1 = np.max(self.thetax) - np.min(self.thetax)
        self.size2 = np.max(self.thetay) - np.min(self.thetay)
        self.nray1 = len(self.thetax)
        self.nray2 = len(self.thetay)

        self.pixel_scale = self.thetax[1]-self.thetax[0]
        self.conv_fact_time = \
                ((1. + self.zl) / c.to(u.km / u.s) *
                 (self.dl * self.ds / self.dls).to(u.km)).to(u.d) * \
                (np.pi / 180.0 / 3600.) ** 2 
        
    def tancl(self):
        lambdat=1.0-self.ka-np.sqrt(self.g1*self.g1+self.g2*self.g2)
        cl = measure.find_contours(lambdat, 0.0)
        for i in range(len(cl)):
            cl[i][:,1] = cl[i][:,1]*self.pixel_scale - self.size1/2.0
            cl[i][:,0] = cl[i][:,0]*self.pixel_scale - self.size2/2.0
        return cl
    
    def radcl(self):
        lambdar=1.0-self.ka+np.sqrt(self.g1*self.g1+self.g2*self.g2)
        cl = measure.find_contours(lambdar, 0.0)
        for i in range(len(cl)):
            cl[i][:,1] = cl[i][:,1]*self.pixel_scale - self.size1/2.0
            cl[i][:,0] = cl[i][:,0]*self.pixel_scale - self.size2/2.0
        return cl

    def crit2cau(self,cl):
        thetac2, thetac1 = np.array(cl[:,0]), np.array(cl[:,1])
        #thetac1_ = ((thetac1+self.size1/2.0)/self.pixel_scale).astype(int)
        #thetac2_ = ((thetac2+self.size1/2.0)/self.pixel_scale).astype(int)

        #ac1 = map_coordinates(self.a1, [[thetac2_], [thetac1_]], order=1, prefilter=True)
        #ac2 = map_coordinates(self.a2, [[thetac2_], [thetac1_]], order=1, prefilter=True)

        a1, a2 = self.angle(thetac1,thetac2)
        betac1 = thetac1 - a1#ac1[0]
        betac2 = thetac2 - a2#ac2[0]

        cau = np.zeros(np.array(cl).shape)
        cau[:,0], cau[:,1] = betac2, betac1
        return cau


class sie(genlen):

    def __init__(self, co, **kwargs):
        self.computed_potential = False
        if ('zl' in kwargs):
            self.zl = kwargs['zl']
        else:
            self.zl = 0.3

        if ('zs' in kwargs):
            self.zs = kwargs['zs']
        else:
            self.zs = 1.5

        if ('theta_c' in kwargs):
            self.theta_c = kwargs['theta_c']
        else:
            self.theta_c = 0.0

        if ('pa' in kwargs):
            self.pa = kwargs['pa'] - np.pi/2.
        else:
            self.pa = 0.0 - np.pi/2.

        if ('q' in kwargs):
            self.q = kwargs['q']
        else:
            self.q = 1.0

        if ('sigma0' in kwargs):
            self.sigma0 = kwargs['sigma0']
        else:
            self.sigma0 = 200.0

        if ('x1' in kwargs):
            self.x1 = kwargs['x1']
        else:
            self.x1 = 0.0

        if ('x2' in kwargs):
            self.x2 = kwargs['x2']
        else:
            self.x2 = 0.0

        self.co = co
        self.dl = co.angular_diameter_distance(self.zl)
        self.ds = co.angular_diameter_distance(self.zs)
        self.dls = co.angular_diameter_distance_z1z2(self.zl, self.zs)

    # @property
    def bsie(self):
        conv = 180.0 / np.pi * 3600.0  # radians to arcsec
        return conv * 4.0 * np.pi * self.sigma0 ** 2 / const.c.to('km/s').value ** 2 * self.dls.value / self.ds.value / np.sqrt(
            self.q)

    def sfunc(self):
        return (self.theta_c / np.sqrt(self.q))


    def kappa(self, theta1__, theta2__):
        theta1 = (theta1__ - self.x1)
        theta2 = (theta2__ - self.x2)
        theta1_ = theta1 * np.sin(self.pa) - theta2 * np.cos(self.pa)
        theta2_ = theta1 * np.cos(self.pa) + theta2 * np.sin(self.pa)
        kappa_ = self.bsie() / 2 * (1.0 / np.sqrt(self.sfunc() ** 2 + theta1_ ** 2 + theta2_ ** 2 / self.q ** 2))
        return kappa_


    def angle(self, theta1__, theta2__):
        theta1 = theta1__ - self.x1
        theta2 = theta2__ - self.x2
        theta1_ = theta1 * np.sin(self.pa) - theta2 * np.cos(self.pa)
        theta2_ = theta1 * np.cos(self.pa) + theta2 * np.sin(self.pa)
        psi = np.sqrt(self.q ** 2 * (self.sfunc() ** 2 + theta1_ ** 2) + theta2_ ** 2)
        if (self.q < 1):
            alphax = self.bsie() * \
                     self.q / np.sqrt(1 - self.q ** 2) * \
                     np.arctan(np.sqrt(1 - self.q ** 2) * theta1_ / (psi + self.sfunc()))
            alphay = self.bsie() * \
                     self.q / np.sqrt(1 - self.q ** 2) * \
                     np.arctanh(np.sqrt(1 - self.q ** 2) * theta2_ / (psi + self.q ** 2 * self.sfunc()))
        elif (self.q == 1):
            alphax = self.bsie() * theta1_ / (psi + self.sfunc())
            alphay = self.bsie() * theta2_ / (psi + self.sfunc())
        else:
            print('q cannot be larger than 1')

        alphax_ = alphax * np.sin(self.pa) + alphay * np.cos(self.pa)
        alphay_ = -alphax * np.cos(self.pa) + alphay * np.sin(self.pa)
        return (alphax_, alphay_)


    def potential(self, theta1__, theta2__):
        theta1 = theta1__ - self.x1
        theta2 = theta2__ - self.x2
        theta1_ = theta1 * np.sin(self.pa) - theta2 * np.cos(self.pa)
        theta2_ = theta1 * np.cos(self.pa) + theta2 * np.sin(self.pa)
        psi = np.sqrt(self.q ** 2 * (self.sfunc() ** 2 + theta1_ ** 2) + theta2_ ** 2)
        if (self.q < 1):
            alphax = self.bsie() * \
                     self.q / np.sqrt(1 - self.q ** 2) * \
                     np.arctan(np.sqrt(1 - self.q ** 2) * theta1_ / (psi + self.sfunc()))
            alphay = self.bsie() * \
                     self.q / np.sqrt(1 - self.q ** 2) * \
                     np.arctanh(np.sqrt(1 - self.q ** 2) * theta2_ / (psi + self.q ** 2 * self.sfunc()))
        elif (self.q == 1):
            alphax = self.bsie() * theta1_ / (psi + self.sfunc())
            alphay = self.bsie() * theta2_ / (psi + self.sfunc())
        else:
            print('q cannot be larger than 1')
        if (np.abs(self.theta_c)>0.0):
            pot = theta1_*alphax+ \
                theta2_*alphay+ \
                self.bsie()*self.q*self.sfunc()*np.log((1.+self.q)*self.sfunc()/
                                                     np.sqrt((psi+self.sfunc())**2+(1.-self.q**2)*
                                                             theta1_**2))
        else:
            pot = theta1_*alphax+theta2_*alphay
        return pot


    def psi11(self, theta1_, theta2_):
        psi = np.sqrt(self.q ** 2 * (self.sfunc() ** 2 + theta1_ ** 2) + theta2_ ** 2)
        den = (1.0 + self.q ** 2) * self.sfunc() ** 2 + 2.0 * psi * self.sfunc() + theta1_ ** 2 + theta2_ ** 2
        psi11_ = self.bsie() * self.q / psi * (
                self.q ** 2 * self.sfunc() ** 2 + theta2_ ** 2 + self.sfunc() * psi) / den
        return (psi11_)


    def psi22(self, theta1_, theta2_):
        psi = np.sqrt(self.q ** 2 * (self.sfunc() ** 2 + theta1_ ** 2) + theta2_ ** 2)
        den = (1.0 + self.q ** 2) * self.sfunc() ** 2 + 2.0 * psi * self.sfunc() + theta1_ ** 2 + theta2_ ** 2
        psi22_ = self.bsie() * self.q / psi * (self.sfunc() ** 2 + theta1_ ** 2 + self.sfunc() * psi) / den
        return (psi22_)


    def psi12(self, theta1_, theta2_):
        psi = np.sqrt(self.q ** 2 * (self.sfunc() ** 2 + theta1_ ** 2) + theta2_ ** 2)
        den = (1.0 + self.q ** 2) * self.sfunc() ** 2 + 2.0 * psi * self.sfunc() + theta1_ ** 2 + theta2_ ** 2
        psi12_ = -self.bsie() * self.q / psi * (theta1_ * theta2_) / den
        return (psi12_)


    def gamma(self, theta1__, theta2__):
        theta1 = theta1__ - self.x1
        theta2 = theta2__ - self.x2
        theta1_ = theta1 * np.sin(self.pa) - theta2 * np.cos(self.pa)
        theta2_ = theta1 * np.cos(self.pa) + theta2 * np.sin(self.pa)
        psi11 = self.psi11(theta1_, theta2_)
        psi22 = self.psi22(theta1_, theta2_)
        psi12 = self.psi12(theta1_, theta2_)
        psi11_ = psi11 * np.sin(self.pa) ** 2 + 2.0 * psi12 * np.sin(self.pa) * np.cos(self.pa) + psi22 * np.cos(
            self.pa) ** 2
        psi22_ = psi11 * np.cos(self.pa) ** 2 - 2.0 * psi12 * np.sin(self.pa) * np.cos(self.pa) + \
                 psi22 * np.sin(self.pa) ** 2
        psi12_ = -psi11 * np.sin(self.pa) * np.cos(self.pa) + psi12 * (np.sin(self.pa) ** 2 - np.cos(self.pa) ** 2) + \
                 psi22 * np.sin(self.pa) * np.cos(self.pa)

        gammax = 0.5 * (psi11_ - psi22_)
        gammay = psi12_

        return (gammax, gammay)

class piemd(genlen):

    def __init__(self, co, **kwargs):
        self.computed_potential = False
        if ('zl' in kwargs):
            self.zl = kwargs['zl']
        else:
            self.zl = 0.3

        if ('zs' in kwargs):
            self.zs = kwargs['zs']
        else:
            self.zs = 1.5

        self.co=co
        self.dl = co.angular_diameter_distance(self.zl)
        self.ds = co.angular_diameter_distance(self.zs)
        self.dls = co.angular_diameter_distance_z1z2(self.zl, self.zs)

        if ('theta_c' in kwargs):
            self.theta_c = kwargs['theta_c']
        else:
            self.theta_c = 0.0

        if ('pa' in kwargs):
            self.pa = kwargs['pa']
        else:
            self.pa = 0.0

        if ('q' in kwargs):
            self.q = kwargs['q']
        else:
            self.q = 1.0

        if ('sigma0' in kwargs):
            self.sigma0 = kwargs['sigma0']
        else:
            self.sigma0 = 200.0

        if ('x1' in kwargs):
            self.x1 = kwargs['x1']
        else:
            self.x1 = 0.0

        if ('x2' in kwargs):
            self.x2 = kwargs['x2']
        else:
            self.x2 = 0.0

        if ('theta_t' in kwargs):
            self.theta_t = kwargs['theta_t']
        else:
            self.theta_t = self.rt()
            self.theta_t = self.theta_t* 1e-3 / self.dl.value * \
                           180.0 / np.pi * 3600.0



        kwargs1 = {'zl': self.zl,
                   'zs': self.zs,
                   'sigma0': self.sigma0,
                   'q': self.q,
                   'theta_c': self.theta_c,
                   'pa': self.pa,
                   'x1': self.x1,
                   'x2': self.x2}
        kwargs2 = {'zl': self.zl,
                   'zs': self.zs,
                   'sigma0': self.sigma0,
                   'q': self.q,
                   'theta_c': self.theta_t,
                   'pa': self.pa,
                   'x1': self.x1,
                   'x2': self.x2}
      
        self.s1 = sie(co, **kwargs1)
        self.s2 = sie(co, **kwargs2)

    def kappa(self, theta1, theta2):
        #isel = (theta1 - self.x1)**2+(theta2 - self.x2)**2 < 5.0*self.theta_t
        kappa_ = self.s1.kappa(theta1, theta2) - self.s2.kappa(theta1, theta2)
        return kappa_

    def potential(self, theta1, theta2):
        pot = self.s1.potential(theta1, theta2) - self.s2.potential(theta1, theta2)
        return pot

    def angle(self, theta1, theta2):
        alphax_1, alphay_1 = self.s1.angle(theta1, theta2)
        alphax_2, alphay_2 = self.s2.angle(theta1, theta2)
        return (alphax_1 - alphax_2, alphay_1 - alphay_2)

    def gamma(self, theta1, theta2):
        gammax_1, gammay_1 = self.s1.gamma(theta1, theta2)
        gammax_2, gammay_2 = self.s2.gamma(theta1, theta2)
        gammax = gammax_1 - gammax_2
        gammay = gammay_1 - gammay_2
        return (gammax, gammay)

    def mass(self):
        GG=const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        return(np.pi*self.sigma0**2/GG *self.theta_t*self.dl.value*np.pi/180.0/3600)

    def m2Dr(self,r):
        GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        rcut=self.theta_t * self.dl.value * np.pi / 180.0 / 3600
        rcore=self.theta_c * self.dl.value * np.pi / 180.0 / 3600
        return (np.pi * self.sigma0 ** 2 / GG * rcut / (rcut-rcore) *
                (np.sqrt(r**2 + rcore**2) - rcore -
                 np.sqrt(r**2 + rcut**2) + rcut))

    def m3Dr(self,r):
        GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        rcut=self.theta_t * self.dl.value * np.pi / 180.0 / 3600
        rcore=self.theta_c * self.dl.value * np.pi / 180.0 / 3600
        return(2 * self.sigma0**2 / GG * rcut / (rcut-rcore) *
               (rcut*np.arctan(r / rcut) - rcore*np.arctan(r / rcore)))

    def vcirc(self,r):
        GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        return(np.sqrt(GG*self.m3Dr(r)/r))

    def rt(self):
        # best fit sigma-rt relation from Bergamini et al. 2019
        r_t = 32.01 * (np.sqrt(3.0 / 2.0) * self.sigma0 / 350.0) ** 2.42
        return r_t

    def density(self,r_in):
        GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        rcut = self.theta_t * self.dl.value * np.pi / 180.0 / 3600
        rcore = self.theta_c * self.dl.value * np.pi / 180.0 / 3600
        r = r_in * self.dl.value * np.pi / 180.0 / 3600
        den = self.sigma0**2.0 / 2.0 / np.pi / GG * (rcut + rcore) / (rcore ** 2 * rcut) / \
              (1.0 + (r/rcore)**2)/(1.0+(r/rcut)**2)
        return den

    def surf_density(self,r_in):
        GG = const.G.to(units.km * units.km / units.s / units.s * units.Mpc / units.Msun).value
        rcut = self.theta_t * self.dl.value * np.pi / 180.0 / 3600
        rcore = self.theta_c * self.dl.value * np.pi / 180.0 / 3600
        r = r_in * self.dl.value * np.pi / 180.0 / 3600
        den = self.sigma0**2.0 / 2.0 / GG * rcut/(rcut - rcore) * \
              (1.0/(rcore**2 + r**2)-1.0/(rcut**2+r**2))
        return den

class td_app(object):

    def __init__(self,co,theta,image=None,**kwargs):

        if image == None:
            # create a plot with two panels only
            self.fig,ax =plt.subplots(1,2,figsize=(10,5))
            fontsize = 15

            lens = sie(co,**kwargs)
            lens.setGrid(theta=theta)
            tancl = lens.tancl()
            radcl = lens.radcl()

            beta1 = 0.0
            beta2 = 0.0

            self.update_td_plot(beta1,beta2,ax,lens,tancl,radcl)

            self.cid = self.fig.canvas.mpl_connect('button_press_event', self.mouse_event)
            for i in range(2):
                ax[i].set_aspect('equal')
                ax[i].xaxis.set_tick_params(labelsize=fontsize)
                ax[i].yaxis.set_tick_params(labelsize=fontsize)
            ax[0].set_xlabel(r'$\beta_1$',fontsize=fontsize)
            ax[0].set_ylabel(r'$\beta_2$',fontsize=fontsize)
            ax[1].set_xlabel(r'$\theta_1$',fontsize=fontsize)
            ax[1].set_ylabel(r'$\theta_2$',fontsize=fontsize)  
        else:
            print ('case2')

    def mouse_event(self,ax,event):
        #print('x: {} and y: {}'.format(event.xdata, event.ydata))
        if event.inaxes not in [ax[0]]:
            print ("Not a clickable region!")
            return

        self.update_td_plot(event.xdata,event.ydata,ax,lens,tancl,radcl)

    def update_td_plot(self,beta1, beta2, ax,lens,tancl,radcl):

        geomtd = self.geom_tdelay_(lens.theta1,lens.theta2,beta1=beta1,beta2=beta2)
        gravtd = - lens.pot

        FOV= lens.theta1.max()-lens.theta1.min()


        td  = (geomtd + gravtd)*lens.conv_fact_time.value

        # define some contour levels
        max_td = 1.5*((-lens.pot).max()-td.min())

        if max_td>0:
            levels = np.linspace(0,max_td,30)
        else:
            levels=30

        ax[1].contourf(td-td.min(),levels=levels,cmap=cm.coolwarm,extent=[-FOV/2.,FOV/2.,-FOV/2.,FOV/2.])
        ax[1].contour(td-td.min(),levels=levels,extent=[-FOV/2.,FOV/2.,-FOV/2.,FOV/2.],colors='white')
        for cl in tancl:
            cau = lens.crit2cau(cl)
            thetac2, thetac1 = cl[:,0], cl[:,1]
            betac2, betac1 = cau[:,0], cau[:,1]
            ax[1].plot(thetac1,thetac2,'--',color='black')
            ax[0].plot(betac1,betac2,'-',color='black')

        for cl in radcl:
            cau = lens.crit2cau(cl)
            thetac2, thetac1 = cl[:,0], cl[:,1]
            betac2, betac1 = cau[:,0], cau[:,1]
            ax[1].plot(thetac1,thetac2,'--',color='black')
            ax[0].plot(betac1,betac2,'-',color='black')

        ax[0].plot(beta1,beta2,'o',ms=5,color='red')

    def geom_tdelay_(self,theta1,theta2,beta1=0.0,beta2=0.0):
        """
        Function to calculate the geometrical time delay:
        inputs: 
        - theta1, theta2 :: mesh of coordinates where the time delay will be evaluated
        - beta1, beta2 :: unlensed source coordinates
        output: 
        - geometrical time delay surface  
        """
        return (0.5*((theta1-beta1)**2+(theta2-beta2)**2))

class test():
    def __init__(self):
        self.fig, ax = plt.subplots()
        self.cid_abc = self.fig.canvas.mpl_connect("button_press_event",self.abc)
    def abc(self, event):
        print("Yes")