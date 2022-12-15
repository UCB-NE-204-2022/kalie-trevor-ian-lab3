import numpy as np

class charge_transport_model():
    '''
    Class for charge transport model calculations
    '''
    def __init__(self):
        self.detector_width = 10 #mm
        self.detector_height = 10.0 # mm
        self.pixel_size_mm = .125 # mm
        self.pixel_pitch_mm = 1
        
        self.cathode_contact_bias = -1000.0
        self.anode_contact_bias = 0.0

        self.charge_density_zero = -1.0 #TODO - change
        self.charge_density_gradient = 0.1 #TODO - change
        
        # build geometry map first thing
        self.build_geom_map()

    def build_geom_map(self):
        '''
        build a map of detector geometry, designating pixels based on their nature
        0 = CZT
        1 = cathode face
        2 = guard ring
        3 = gap 
        4 = anode 4 (right)
        5 = anode 5 (middle)
        6 = anode 6 (left)
        
        '''
        # build (uniform) 2D grids for plotting and solving (x=width, y=height (cathode to anode))
        self.x_range = np.arange(0, self.detector_width+self.pixel_size_mm, self.pixel_size_mm)
        self.y_range = np.arange(0, self.detector_height+self.pixel_size_mm, self.pixel_size_mm)

        # initialize grid for solving for V
        self.N_xelements = np.shape(self.x_range)[0]
        self.N_yelements = np.shape(self.y_range)[0]

        # XY mesh for plotting later
        self.X, self.Y = np.meshgrid(y_range, x_range)
        
        # find anodes
        left_pixel = self.x_range >= 3.375 +.25
        left_pixel *= self.x_range <= 3.375 +.25+.75

        middle_pixel = self.x_range >= 3.375 +.25+.75+.25
        middle_pixel *= self.x_range <= 3.375 +.25+.75 +.25 +.75

        right_pixel = self.x_range >= 3.375 +.25+.75 +.25 +.75 + .25
        right_pixel *= self.x_range <= 3.375 +.25+.75 +.25 +.75 +.25 + .75

        left_pixel_idx = np.argwhere(left_pixel).flatten()
        middle_pixel_idx = np.argwhere(middle_pixel).flatten()
        right_pixel_idx = np.argwhere(right_pixel).flatten()

        # estimated gr width from info doc
        gr_width_est = .75*2+.25*3+.75/2

        gr_left = self.x_range < 3.375
        gr_left *= self.x_range > 3.375-gr_width_est
        gr_right = self.x_range > 3.375 +.25+.75 +.25 +.75 +.25 + .75 + .25
        gr_right *= self.x_range < 3.375 +.25+.75 +.25 +.75 +.25 + .75 + .25 + gr_width_est
        gr_left_idx = np.argwhere(gr_left).flatten()
        gr_right_idx = np.argwhere(gr_right).flatten()
        guard_ring_idx = np.concatenate((gr_left_idx,gr_right_idx))

        # detector all 0
        geom_map = np.zeros((N_xelements, N_yelements), dtype=int)

        # contacts
        geom_map[-1,:] = 1 # cathode
        geom_map[0,guard_ring_idx] = 2 # guard ring
        geom_map[0,right_pixel_idx] = 4 # pixel 4
        geom_map[0,middle_pixel_idx] = 5 # pixel 5
        geom_map[0,left_pixel_idx] = 6 # pixel 6

        # gaps
        gap_idx = np.argwhere(geom_map[0] == 0).flatten()
        geom_map[0,gap_idx] = 3
        # geom_map[0,anode_idx] = 2 # anodes
        self.geom_map = geom_map
        
    def find_WP(self,
        contact,
        max_iters=1000):
        '''
        Find weighting potential of specified contact
        
        contacts: 
        1 = cathode face
        4 = anode 4 (right)
        5 = anode 5 (middle)
        6 = anode 6 (left)
        
        contact: int
            contact number (see map)
        '''
        # set initial conditions
        # set V = 0 for everything except desired contact
        V = np.zeros((self.N_xelements, self.N_yelements), dtype=float)
        # set V = 1 for desired contact
        x, y = np.where(self.geom_map == contact)
        V[x, y] = 1
        
        # do the relaxation
        # set maximum number of iterations
        max_iters = max_iters

        # "over-relaxation" factor to speed up convergence
        t = np.cos(3.14/self.N_xelements) + np.cos(3.14/self.N_yelements)
        w = (8 - np.sqrt(64 - 16*t*t)) / (t*t)

        # initialise arrays which will store the residuals
        R = np.zeros((self.N_xelements,self. N_yelements), dtype=float)
        resid_store = np.zeros(max_iters)

        # perform relaxation...
        resid = 1e6
        iterr = 1
        min_resid = 0.01

        while (iterr < max_iters and resid > min_resid):    
            
            # loop over detector grid points
            for y in range(1, self.N_yelements-1):
                for x in range(0, self.N_xelements-1):
                                
                    # skip pixels with boundary conditions
                    if ((self.geom_map[x,y] == 1) or (self.geom_map[x,y] == 2) or (self.geom_map[x,y] == 3) or (self.geom_map[x,y] == 4) or (self.geom_map[x,y] == 5) or (self.geom_map[x,y] == 6)):
                        continue

                    # should deal with some boundary conditions...
                                        
                    V_local_sum = (V[x+1,y] + V[x,y+1] + V[x-1,y] + V[x,y-1])
                    
                    # update the solution
                    R[x,y] = 0.25*V_local_sum - V[x,y]
                    V[x,y] = V[x,y] + w*R[x,y]
                    
            # calculate the residual and store as a function of iteration number
            resid = abs(np.sum(R))
            resid_store[iterr] = resid
            
            # update iteration counter
            iterr+=1
            
        self.WP = V
        
        # plot relaxation
        plt.figure()
        plt.plot(np.arange(1,iterr), resid_store)
        plt.grid("on")
        plt.xlabel("Iteration Number")
        plt.ylabel("Difference")
        #plt.yscale("log")
        plt.show()
    
    def plot_WP(self):
        plt.figure()
        plt.imshow(self.V,interpolation="None",cmap='jet',vmin=-0.1)
        plt.xticks([])
        plt.yticks([])
        plt.title('Weighting Potential of Anode 5')
        plt.colorbar()
        plt.show()
        
    def plot_WP_slice(self,
        slice_idx):
        '''
        Plot slice of WP
        
        Parameters
        ----------
        slice_idx: int
            which slice to plot
        
        '''
        WPslice = self.WP[:,slice_idx]
        plt.figure()
        plt.plot(self.x_range, WPslice)
        plt.grid("on")
        plt.xlabel("Depth (mm)")
        plt.ylabel("Weighting Potential")
        plt.show()
        
    def drift_particles(self,
        slice_idx,
        depth_mm):
        '''
        Drift particles and return Q in au
        
        Parameters
        ----------
        slice_idx: int
            slice in detector
        depth_mm: num
            interaction depth in mm
            
        Returns
        -------
        Qsignal_h: ndarray
            hole signal
        Qsignal_e: ndarray
            electron signal
        Qsignal: ndarray
            sum of Qh and Qe
        '''
        # Vslice = WPslice
        # simple signal calculation assuming fixed velocity of one pixel per ns for both electrons and holes
        # holes into wp, electrons out
        # could easily vectorize to be faster and more pythonic
        
        # grab slice of weighting potential
        Vslice = self.WP[:,slice_idx]

        z0 = np.int(np.floor(depth_mm / self.pixel_size_mm))

        # number of time steps in signal
        Nt = 150

        # arrays which will store the induced charge signals
        Qh = np.zeros(Nt, dtype=float)
        Qe = np.zeros(Nt, dtype=float)

        # starting positions for electrons and holes
        zh = z0
        ze = z0

        # holes into wp
        t = 0
        for t in range(1, Nt):
            if (zh<=self.N_yelements-1):
                dw = Vslice[zh] - Vslice[zh-1]
                Qh[t] = 1.0*dw    
            elif (zh>self.N_yelements-1):
                continue
            zh = zh+1

        # electrons out of wp
        t = 0
        for t in range(1, Nt):
            if (ze>=0):
                dw = Vslice[ze] - Vslice[ze+1]
                Qe[t] = -1.0*dw
            elif (ze<0):
                continue
            ze = ze-1

        # take cumulative sums
        Qsignal_h = np.cumsum(Qh)
        Qsignal_e = np.cumsum(Qe)
        Qsignal = np.cumsum(Qe + Qh)
        
        # plot results
        # plot
        plt.figure()
        plt.plot(Qsignal_e, 'm', linewidth=1.5,label='Qe')
        plt.plot(Qsignal_h, 'c', linewidth=1.5,label='Qh')
        plt.plot(Qsignal, 'k', linewidth=2,label='Qsignal')
        plt.grid("on")
        #plt.ylim(0,1)
        plt.xlim(0, Nt)
        plt.tick_params(labelbottom="off")
        plt.xlabel("Time")
        plt.ylabel("Charge (au)")
        plt.title('Signal for interaction at 5 mm depth')
        plt.legend()
        plt.show()
        return Qsignal_h, Qsignal_e, Qsignal
