from astropy import constants as const
from astropy.constants import c
from astropy import units as u
import astropy.units as units
import numpy as np
from skimage import measure
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.modeling.functional_models import Sersic2D
from scipy.optimize import fsolve, newton,brenth, brentq

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
        #self.conv_fact_time = \
        #        ((1. + self.zl) / c.to(u.km / u.s) *
        #         (self.dl * self.ds / self.dls).to(u.km)).to(u.d) * \
        #        (np.pi / 180.0 / 3600.) ** 2 
        
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

    def combinewith(self,gl):

        if np.round(gl.pixel_scale,6) != np.round(self.pixel_scale,6)\
                or gl.size1 != self.size1 \
                or gl.size2 != self.size2:
            raise Exception('Incompatible sizes of deflectors (conbinewith)')
        self.ka=self.ka+gl.ka
        self.g1=self.g1+gl.g1
        self.g2=self.g2+gl.g2
        self.a1=self.a1+gl.a1
        self.a2=self.a2+gl.a2


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

        if self.theta_c < 1e-4:
            self.lens_type = 'SIE'
        else:
            self.lens_type = 'NIE'   

        self.co = co
        self.dl = co.angular_diameter_distance(self.zl)
        self.ds = co.angular_diameter_distance(self.zs)
        self.dls = co.angular_diameter_distance_z1z2(self.zl, self.zs)
        self.conv_fact_time = \
            ((1. + self.zl) / c.to(u.km / u.s) *
            (self.dl * self.ds / self.dls).to(u.km)).to(u.d) * \
            (np.pi / 180.0 / 3600.) ** 2 

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

    def mu(self, theta1__, theta2__):
        kappa = self.kappa(theta1__, theta2__)
        gammax,gammay = self.gamma(theta1__, theta2__)
        gamma=np.sqrt(gammax**2+gammay**2)
        mu = 1.0/abs((1-kappa)**2-gamma**2)
        return mu


class piemd(genlen):

    def __init__(self, co, **kwargs):
        self.computed_potential = False
        self.lens_type = 'PIEMD'
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
        self.conv_fact_time = \
            ((1. + self.zl) / c.to(u.km / u.s) *
            (self.dl * self.ds / self.dls).to(u.km)).to(u.d) * \
            (np.pi / 180.0 / 3600.) ** 2         

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


class gensrc(object):
    def __init__(self) -> object:
        self.initialized=True

    def ray_trace(self):
        px = self.df.pixel_scale

        x1pix = (self.x1 - self.df.thetax[0]) / px
        x2pix = (self.x2 - self.df.thetay[0]) / px

        if len(x1pix.shape) > 1:
            x1pix[:, -1] = np.round(x1pix[:, -1], 0)
            x2pix[-1, :] = np.round(x2pix[-1, :], 0)
            x1pix[:, 0] = np.round(x1pix[:, 0], 0)
            x2pix[0, :] = np.round(x2pix[0, :], 0)
        else:
            x1pix[-1] = np.round(x1pix[-1], 0)
            x2pix[-1] = np.round(x2pix[-1], 0)
            x1pix[0] = np.round(x1pix[0], 0)
            x2pix[0] = np.round(x2pix[0], 0)

        a1 = map_coordinates(self.df.a1,
                             [x2pix, x1pix], order=1, prefilter=True)
        a2 = map_coordinates(self.df.a2,
                             [x2pix, x1pix], order=1, prefilter=True)

        y1 = (self.x1 - a1 * self.rescf)  # y1 coordinates on the source plane
        y2 = (self.x2 - a2 * self.rescf)  # y2 coordinates on the source plane

        return (y1, y2)


class pointsrc(gensrc):

    def __init__(self, size=100.0, sizex=None, sizey=None, Npix=100, gl=None,
                 save_unlensed=False, fast=False, **kwargs):

        if ('ys1' in kwargs):
            self.ys1 = kwargs['ys1']
        else:
            self.ys1 = 0.0

        if ('ys2' in kwargs):
            self.ys2 = kwargs['ys2']
        else:
            self.ys2 = 0.0

        if ('flux' in kwargs):
            self.flux = kwargs['flux']
        else:
            self.flux = 100.0

        if ('zs' in kwargs):
            self.zs = kwargs['zs']
        else:
            self.zs = 1.0

        self.rescf = 1.0
        if gl != None:
            if self.zs != gl.zs:
                if self.zs > gl.zl:
                    ds = gl.co.angular_diameter_distance(self.zs).value
                    dls = gl.co.angular_diameter_distance_z1z2(gl.zl,self.zs).value
                    self.rescf=dls/ds*gl.ds/gl.dls
                else:
                    self.rescf = 0.0

        self.N = Npix
        self.df = gl

        if not fast:
            # define the pixel coordinates
            if sizex == None or sizey == None:
                self.size = float(size)
                pcx = np.linspace(-self.size / 2.0, self.size / 2.0, self.N)
                pcy = np.linspace(-self.size / 2.0, self.size / 2.0, self.N)
                self.center_frame = [0.,0.]
            else:
                pcx = np.linspace(sizex[0], sizex[1], self.N)
                pcy = np.linspace(sizey[0], sizey[1], self.N)
                self.size=sizex[1]-sizex[0]
                self.center_frame = [(sizex[0] + sizex[1]) / 2.0, (sizey[0] + sizey[1]) / 2.0]
        

            self.x1, self.x2 = np.meshgrid(pcx, pcy)

            self.pixel = self.size / self.N

            if self.df == None: # NO LENS
                self.image = self.brightness(self.ys1,self.ys2)
                if(save_unlensed):
                    self.image_unlensed = self.image
            else:               # LENS
                if(save_unlensed):
                    self.image_unlensed = self.brightness(self.ys1,self.ys2)
                self.image = np.zeros((self.N, self.N))
                self.xi1, self.xi2, self.mui, self.tdi = self.find_images()

                for i in range(len(self.xi1)):
                    image_tmp = self.brightness(self.xi1[i],self.xi2[i], self.mui[i])
                    self.image=self.image+image_tmp.copy()
        else:
            self.xi1, self.xi2, self.mui, self.tdi = self.find_images()


    def brightness(self,ys1,ys2,mu=1.0):
        # convert image positions with respect to the deflector center into 
        # image positions with respect to the image center
        #print (self.center_frame)
        ys1 = ys1 - self.center_frame[0]
        ys2 = ys2 - self.center_frame[1]

        pix = self.size / self.N   # LB: this is already defined above as self.pixel...

        px = int(ys1/pix + self.N/2.)
        py = int(ys2/pix + self.N/2.)
        brightness=np.zeros((self.N,self.N))
        if ((px >= 0) & (px<self.N) & (py >= 0) & (py < self.N)):
            brightness[py,px]=self.flux*mu
        return (brightness)


    '''
    Find the images of a source at (ys1,ys2) by mapping triangles on the lens plane into
    triangles in the source plane. Then search for the triangles which contain the source. 
    The image position is then computed by weighing with the distance from the vertices of the 
    triangle on the lens plane
    '''

    def find_images(self):

        if (self.df.lens_type == 'SIE'):
            """
            If the lens is singular, use a faster method to find the multiple images than triangle mapping
            """
            x,phi = self.phi_ima(checkplot=False,verbose=False)
            xi=x*np.cos(phi)
            yi=x*np.sin(phi)
            mui = self.df.mu(xi,yi)
            poti = self.df.potential(xi,yi)
            tdi = (0.5*((self.ys1-xi)**2+(self.ys2-yi)**2)-poti)*self.df.conv_fact_time.value
            tdi = tdi-tdi.min()
            return xi, yi, mui, tdi

        # map the source position in pixels onto the deflector grid
        y1s = self.ys1/self.df.pixel_scale + len(self.df.thetax) / 2.0
        y2s = self.ys2/self.df.pixel_scale + len(self.df.thetay) / 2.0


        # ray-trace the deflector grid onto the source plane
        y1 = self.df.theta1 - self.df.a1*self.rescf
        y2 = self.df.theta2 - self.df.a2*self.rescf

        # convert to pixel units
        xray = y1.copy() / self.df.pixel_scale + len(self.df.thetax) / 2.0
        yray = y2.copy() / self.df.pixel_scale + len(self.df.thetay) / 2.0

        # shift the maps by one pixel
        xray1 = np.roll(xray,   1, axis=1)
        xray2 = np.roll(xray1,  1, axis=0)
        xray3 = np.roll(xray2, -1, axis=1)
        yray1 = np.roll(yray,   1, axis=1)
        yray2 = np.roll(yray1,  1, axis=0)
        yray3 = np.roll(yray2, -1, axis=1)

        """
        For each pixel on the LENS plane, build two triangles. By means of 
        ray-tracing these are mapped onto the source plane into other two 
        triangles. Compute the distances of the vertices of the triangles on 
        the SOURCE plane from the source and check using cross-products if the
        source is inside one of the two triangles.
        """
        # l1=((yray1-yray2)*(ys1-xray2)+(xray2-xray1)*(ys2-yray2))/((yray1-yray2)*(xray-xray2)+(xray2-xray1)*(yray-yray2))

        x1 = y1s - xray
        y1 = y2s - yray

        x2 = y1s - xray1
        y2 = y2s - yray1

        x3 = y1s - xray2
        y3 = y2s - yray2

        x4 = y1s - xray3
        y4 = y2s - yray3

        prod12 = x1 * y2 - x2 * y1
        prod23 = x2 * y3 - x3 * y2
        prod31 = x3 * y1 - x1 * y3
        prod13 = -prod31
        prod34 = x3 * y4 - x4 * y3
        prod41 = x4 * y1 - x1 * y4

        image = np.zeros(xray.shape)
        image[((np.sign(prod12) == np.sign(prod23)) & (np.sign(prod23) == np.sign(prod31)))] = 1
        image[((np.sign(prod13) == np.sign(prod34)) & (np.sign(prod34) == np.sign(prod41)))] = 2

        # In the following, the choices 'image == 1' and 'image == 2' stand for
        # upper and lower triangles (or viceversa).

        # first kind of images (first triangle)
        images1 = np.argwhere(image == 1)
        xi_images_ = images1[:, 1]
        yi_images_ = images1[:, 0]
        xi_images = xi_images_[(xi_images_ > 0) & (yi_images_ > 0)]
        yi_images = yi_images_[(xi_images_ > 0) & (yi_images_ > 0)]

        # compute the weights
        w = np.array([1. / np.sqrt(x1[xi_images, yi_images] ** 2 + y1[xi_images, yi_images] ** 2),
                      1. / np.sqrt(x2[xi_images, yi_images] ** 2 + y2[xi_images, yi_images] ** 2),
                      1. / np.sqrt(x3[xi_images, yi_images] ** 2 + y3[xi_images, yi_images] ** 2)])
        xif1, yif1 = self.refineImagePositions(xi_images, yi_images, w, 1)

        # second kind of images
        images1 = np.argwhere(image == 2)
        xi_images_ = images1[:, 1]
        yi_images_ = images1[:, 0]
        xi_images = xi_images_[(xi_images_ > 0) & (yi_images_ > 0)]
        yi_images = yi_images_[(xi_images_ > 0) & (yi_images_ > 0)]

        # compute the weights
        w = np.array([1. / np.sqrt(x1[xi_images, yi_images] ** 2 + y1[xi_images, yi_images] ** 2),
                      1. / np.sqrt(x3[xi_images, yi_images] ** 2 + y3[xi_images, yi_images] ** 2),
                      1. / np.sqrt(x4[xi_images, yi_images] ** 2 + y4[xi_images, yi_images] ** 2)])
        xif2, yif2 = self.refineImagePositions(xi_images, yi_images, w, 2)

        xi = np.concatenate([xif1, xif2])
        yi = np.concatenate([yif1, yif2])

        mui=self.mu_image(xi,yi)
        poti=self.pot_image(xi,yi)

        xi = (xi - 1 - len(self.df.thetax) / 2.0) * self.df.pixel_scale
        yi = (yi - 1 - len(self.df.thetay) / 2.0) * self.df.pixel_scale

        tdi=(0.5*((self.ys1-xi)**2+(self.ys2-yi)**2)-poti)*self.df.conv_fact_time.value

        isel = mui > 1e-2
        tdi[isel] = tdi[isel]-tdi[isel].min()
        return (xi[isel], yi[isel], mui[isel], tdi[isel])

    def refineImagePositions(self, x, y, w, typ):
        """Image positions are computed as weighted means of the positions
        of the triangle vertices. The weights are the distances between the
        vertices mapped onto the source plane, and the source position."""

        if (typ == 2):
            xp = np.array([x, x + 1, x + 1])
            yp = np.array([y, y, y + 1])
        else:
            xp = np.array([x, x + 1, x])
            yp = np.array([y, y + 1, y + 1])
        xi = np.zeros(x.size)
        yi = np.zeros(y.size)
        for i in range(x.size):
            xi[i] = (xp[:, i] / w[:, i]).sum() / (1. / w[:, i]).sum()
            yi[i] = (yp[:, i] / w[:, i]).sum() / (1. / w[:, i]).sum()
        return (xi, yi)

    def mu_image(self,xi1,xi2):
        mu = map_coordinates((1.0-self.df.ka)**2-self.df.g1**2-self.df.g2**2,
                             [xi2-1, xi1-1], order=1, prefilter=True)
        return(np.abs(1./mu))

    def pot_image(self,xi1,xi2):
        pot = map_coordinates(self.df.pot,
                             [xi2-1, xi1-1], order=1, prefilter=True)
        return pot


    #### Functions to be used with SIE models
    def x_ima(self,phi):
        x=self.ys1*np.cos(phi)+self.ys2*np.sin(phi)+(self.psi_tilde(phi+self.pa))
        return x

    def psi_tilde(self,phi):
        if (self.df.q < 1.0):
            fp=np.sqrt(1.0-self.df.q**2)
            return np.sqrt(self.df.q)/fp*(np.sin(phi-self.df.pa)*np.arcsin(fp*np.sin(phi-self.df.pa))+
                                       np.cos(phi-self.df.pa)*np.arcsinh(fp/self.df.q*np.cos(phi-self.df.pa)))
        else: 
            return(1.0)

    def psi(self,x,phi):
        psi=x*self.psi_tilde(phi)
        return psi
    
    def x_ima(self,y1,y2,phi):
        x=y1*np.cos(phi)+y2*np.sin(phi)+(self.psi_tilde(phi+self.df.pa))
        return x

    def alpha(self,phi):
        fp=np.sqrt(1.0-self.df.q**2)
        a1=np.sqrt(self.df.q)/fp*np.arcsinh(fp/self.df.q*np.cos(phi))
        a2=np.sqrt(self.df.q)/fp*np.arcsin(fp*np.sin(phi))
        return a1,a2
    
    def phi_ima(self,checkplot=True,verbose=True):
        y1_ = self.ys1 * np.cos(self.df.pa) + self.ys2 * np.sin(self.df.pa)
        y2_ = - self.ys1 * np.sin(self.df.pa) + self.ys2 * np.cos(self.df.pa)

        y1_= y1_/self.df.bsie()/np.sqrt(self.df.q)
        y2_= y2_/self.df.bsie()/np.sqrt(self.df.q)
    
        def phi_func(phi):
            a1,a2=self.alpha(phi)
            func=(y1_+a1)*np.sin(phi)-(y2_+a2)*np.cos(phi)
            return func

        U=np.linspace(0.,2.0*np.pi+0.001,100)
        c = phi_func(U)
        s = np.sign(c)
        phi=[]
        xphi=[]
        for i in range(len(U)-1):
            if s[i] + s[i+1] == 0: # opposite signs
                u = brentq(phi_func, U[i], U[i+1])
                z = phi_func(u)
                if np.isnan(z) or abs(z) > 1e-3:
                    continue
                x=self.x_ima(y1_,y2_,u)
                if (x>0):
                    phi.append(u)
                    xphi.append(x)
                if (verbose):
                    print('found zero at {}'.format(u))
                    if (x<0):
                        print ('discarded because x is negative ({})'.format(x))
                    else:
                        print ('accepted because x is positive ({})'.format(x))
                        
        xphi=np.array(xphi)
        phi=np.array(phi)
        if (checkplot):        
            phi_=np.linspace(0.,2.0*np.pi,100)
            ax[0].plot(phi_,phi_func(phi_),label=r'$F(\varphi)$')
            ax[0].plot(phi_,self.x_ima(y1,y2,phi_),label=r'$x(\varphi)$')
            #ax[0].plot(phi_,psi_tilde(phi_,f)-1)
            ax[0].plot(phi,phi_func(phi),'o',markersize=8)
            ax[0].set_xlabel(r'$\varphi$',fontsize=20)
            ax[0].set_ylabel(r'$F(\varphi),x(\varphi)$',fontsize=20)
            ax[0].legend(fontsize=16)
    
        return(xphi*self.df.bsie()*np.sqrt(self.df.q),phi+self.df.pa)

class sersic(gensrc):

    def __init__(self, size=100.0, Npix=100, gl=None, sizex=None, sizey=None, pcx=None, pcy=None, mask=None, save_unlensed=False, rmaxf=100, **kwargs):

        if ('n' in kwargs):
            self.n = kwargs['n']
        else:
            self.n = 4

        if ('re' in kwargs):
            self.re = kwargs['re']
        else:
            self.re = 5.0

        if ('q' in kwargs):
            self.q = kwargs['q']
        else:
            self.q = 1.0

        if ('pa' in kwargs):
            self.pa = kwargs['pa']
        else:
            self.pa = 0.0

        if ('ys1' in kwargs):
            self.ys1 = kwargs['ys1']
        else:
            self.ys1 = 0.0

        if ('ys2' in kwargs):
            self.ys2 = kwargs['ys2']
        else:
            self.ys2 = 0.0

        if ('c' in kwargs):
            self.c = kwargs['c']
        else:
            self.c = 0.0

        if ('Ie' in kwargs):
            self.Ie = kwargs['Ie']
        else:
            self.Ie = 100.0

        if ('zs' in kwargs):
            self.zs = kwargs['zs']
        else:
            self.zs = 1.0


        if gl != None:
            if self.zs != gl.zs:
                if self.zs > gl.zl:
                    ds = gl.co.angular_diameter_distance(self.zs).value
                    dls = gl.co.angular_diameter_distance_z1z2(gl.zl,self.zs).value
                    self.rescf=dls/ds*gl.ds/gl.dls
                else:
                    self.rescf = 0.0
            else:
                self.rescf=1.0
        else:
            self.rescf=1.0

        self.df = gl

        self.mask = mask
        self.save_unlensed = save_unlensed

        # define the pixel coordinates
        if pcx is not None or pcy is not None:
            self.Nx = len(pcx)
            self.Ny = len(pcx)
            self.sizex = np.max(pcx)-np.min(pcx)
            self.sizey = np.max(pcy)-np.min(pcy)
            if np.round((pcx[1]-pcx[0]),6) != np.round((pcy[1]-pcy[0]),6):
                raise Exception('Pixels of pcx and pcy must have the same dimension')

        elif sizex is not None or sizey is not None:
            self.N = Npix
            pcx = np.linspace(sizex[0], sizex[1], self.N)
            pcy = np.linspace(sizey[0], sizey[1], self.N)
            self.Nx = len(pcx)
            self.Ny = len(pcx)
            self.sizex = np.max(pcx)-np.min(pcx)
            self.sizey = np.max(pcy)-np.min(pcy)

        else:
            self.size = size
            self.N = Npix
            pcx = np.linspace(-self.size / 2.0, self.size / 2.0, self.N)
            pcy = np.linspace(-self.size / 2.0, self.size / 2.0, self.N)
            self.Nx = Npix
            self.Ny = Npix
            self.sizex = float(size)
            self.sizey = float(size)

        self.x1, self.x2 = np.meshgrid(pcx, pcy)

        if mask is not None:
            image_mask = np.full(self.x1.shape,np.nan)
            self.x1, self.x2 = self.x1[self.mask], self.x2[self.mask]

        if self.df != None:
            y1, y2 = self.ray_trace()
        else:
            y1, y2 = self.x1, self.x2

        self.y1 = y1
        self.y2 = y2

        self.image = self.brightness(y1, y2, rmaxf)
        if (save_unlensed):
            self.image_unlensed = self.brightness(self.x1, self.x2, rmaxf=rmaxf)

        if self.mask is not None:
            image_mask[self.mask] = self.image
            self.image = image_mask

    def brightness(self, y1, y2, rmaxf):
        px = self.sizex / (self.Nx - 1)

        brightness = np.zeros_like(y1)
        isel = (y1 < rmaxf*self.re + self.ys1) & (y2 < rmaxf*self.re + self.ys2)

        s = Sersic2D(amplitude=self.Ie, r_eff=self.re, n=self.n, x_0=self.ys1, y_0=self.ys2,
                     ellip=np.sqrt(1-self.q**2), theta=self.pa + np.pi/2)

        brightness[isel] = s(y1[isel], y2[isel])*px*px

        return brightness