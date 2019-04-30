import matplotlib.pyplot as plt
import numpy as np
import sympy

masses = {
    'Al':26.98,
    'Ga':69.72,
    'In':114.82,
    'Zn':65.38,
    'Cd':112.41,
    'N':14.00,
    'O':16.00,
}

file = 'static11.txt'
with open(file) as f:
    txt = f.read().split('\n')
    def split_list(big_list, x):
        return [big_list[i:i+x] for i in range(0, len(big_list), x)]

    parents = txt[0].split(' ')
    children = [t.split(' ') for t in txt[1:6]]
    sigma11s = split_list([float(f) for f in txt[6].split(' ')], 2)
    A11s = split_list([float(f) for f in txt[7].split(' ')], 2)
    B11s = split_list([float(f) for f in txt[8].split(' ')], 2)
    
file = 'static33.txt'
with open(file) as f:
    txt = f.read().split('\n')
    def split_list(big_list, x):
        return [big_list[i:i+x] for i in range(0, len(big_list), x)]

    parents = txt[0].split(' ')
    children = [t.split(' ') for t in txt[1:6]]
    sigma33s = split_list([float(f) for f in txt[6].split(' ')], 2)
    A33s = split_list([float(f) for f in txt[7].split(' ')], 2)
    B33s = split_list([float(f) for f in txt[8].split(' ')], 2)
    
data = [{'material':parent, 'elements':child, 's11':s11, 'a11':a11, 'b11':b11, 's33':s33, 'a33':a33, 'b33':b33} for parent, child, s11, a11, b11, s33, a33, b33 in zip(parents, children, sigma11s, A11s, B11s, sigma33s, A33s, B33s)]

class submaterial():
    def __init__(self, parent=None):
        self.parent = parent
        self.name = None
        self.M = None
        self.sigma = None
        self.A = None
        self.B = None
        
    def __repr__(self,):
        return 'Static Correlation Function for {} in {}'.format(self.name, self.parent)
                                                           
    def static_correlation(self, entry, T):
        '''entry is 0 or 1, for 11 or 33 directions respectively
        '''
        import scipy.constants as con
        import numpy as np
        
        def coth(x):
            result = 1/np.tanh(x)
            #result = (np.exp(x) + np.exp(-x))/(np.exp(x) - np.exp(-x))
            return result
        
        k = con.k
        hbar = con.hbar
        amu = con.atomic_mass
        
        sigma = self.sigma[entry]
        A = self.A[entry]
        B = self.B[entry]
            
        self.exponent = -(T**2)/(sigma**2)
        exponent = self.exponent
        self.pre_N = hbar/(2*k*T)
        pre_N = self.pre_N
        self.N = (A*np.exp(exponent)+B) #N = inside square brackets
        N = self.N
        self.pre = hbar/(2*self.M*amu*N)
        pre = self.pre
        Å = 1e-10
        equation = pre*coth(pre_N*N)/(Å**2)
        return equation

    def u11(self, T):
        return self.static_correlation(0, T)
    
    def u33(self, T):
        return self.static_correlation(1, T)

    def u2(self, T):
        '''Static correlation function for a given temperature (range) T'''
        return (2*self.u11(T) + self.u33(T))/3
    
    def rms(self, T):
        return np.sqrt((self.u11(T) + self.u11(T) + self.u33(T))/3)

class material():
    '''Method for returning the directional and isotropic static correlation functions
    '''
    def __init__(self, data):
        '''Enter the various parameters from the Schowalter (2009) paper here
        '''
        self.data = data        
        self.separate_data()
        
    def __repr__(self,):
        return 'Static Correlation Function for {}'.format(self.name)
        
    def separate_data(self):
        data = self.data
        self.name = data['material']
        
        self.metal = submaterial(self.name)
        self.nonmetal = submaterial(self.name)
        
        self.metal.name = data['elements'][0]
        self.metal.M = masses[self.metal.name]
        self.metal.sigma = (data['s11'][0], data['s33'][0])
        self.metal.A = (data['a11'][0], data['a33'][0])
        self.metal.B = (data['b11'][0], data['b33'][0])
        
        self.nonmetal.name = data['elements'][1]
        self.nonmetal.M = masses[self.nonmetal.name]
        self.nonmetal.sigma = (data['s11'][1], data['s33'][1])
        self.nonmetal.A = (data['a11'][1], data['a33'][1])
        self.nonmetal.B = (data['b11'][1], data['b33'][1])
        
materials = [material(d) for d in data]
AlN = materials[0]
GaN = materials[1]
InN = materials[2]
ZnO = materials[3]
CdO = materials[4]

def trans(T, yvalue):
    res = ax.transAxes.inverted().transform(ax.transData.transform([T, yvalue]))
    return res


def from_AXES(axes):
    return ax.transAxes.transform(axes)

def to_AXES(display):
    return ax.transAxes.inverted().transform(display)

def from_DATA(data):
    return ax.transData.transform(data)

def to_DATA(display):
    return ax.transData.inverted().transform(display)


mat = ZnO
Tmax = 400
RT = 300
LN2 = 100

T = np.arange(0.1,Tmax,0.01)
fig, ax = plt.subplots(1,1, figsize=(8,4))
lns = []

lns.append(ax.plot(T, mat.metal.rms(T), color='red', lw=5, ls='--', label='{} in {}'.format(mat.metal.name, mat.name))[0])
lns.append(ax.plot(T, mat.nonmetal.rms(T), color='green', lw=5, ls='dashed', label='{} in {}'.format(mat.nonmetal.name, mat.name))[0])

lns.append(ax.axvline(RT, ymax=trans(RT, mat.metal.rms(RT))[1], c='black', ls='--'))
lns.append(ax.axvline(LN2, ymax=trans(LN2, mat.nonmetal.rms(LN2))[1], c='black', ls='--'))

#ax.axhline(AlN.nonmetal.rms(LN2), xmax=trans(LN2, AlN.nonmetal.rms(LN2))[0], c='green')
#ax.axhline(mat.metal.rms(LN2), xmax=trans(LN2, mat.metal.rms(LN2))[0], c='red')

ax.annotate('  RT', xy=(RT, to_DATA(from_AXES(to_AXES(from_DATA([0, mat.metal.rms(RT)]))/2))[1]))
ax.annotate('  LN$_2$', xy=(LN2, to_DATA(from_AXES(to_AXES(from_DATA([0, mat.metal.rms(LN2)]))/2))[1]))
plt.legend(loc=2)
plt.xlim(0,Tmax)
plt.title('Temperature dependence of Debye-Waller factor')
plt.xlabel('Temperature (K)')
plt.ylabel(r'RMS Debye Waller Factor (Å)')
#plt.ylabel(r'Root of Isotropic Static Correlation Function (Å)')
fig.savefig('debye_waller.png', dpi=200)