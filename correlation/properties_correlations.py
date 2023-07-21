import numpy as np
from darts.physics import *

class PropertyContainer(property_evaluator_iface):
    def __init__(self, phase_name, component_name, rate_name):
        super().__init__()
        # This class contains all the property evaluators required for simulation
        self.n_phases = len(phase_name)
        self.nc = len(component_name)
        self.component_name = component_name
        self.phase_name = phase_name
        self.rate_name = rate_name

        # Allocate (empty) evaluators
        self.density_ev = []
        self.viscosity_ev = []
        self.watersat_ev = []
        self.rel_perm_ev = []
        self.capillary_ev = []
        self.rock_compress_ev = []

class DensityOil(property_evaluator_iface):
     def __init__(self, pref=1, Bref=1.01, compres=0, surface_oil_dens=849):
        super().__init__()
        self.surf_oil_dens = surface_oil_dens
        self.Pref = pref
        self.Bref = Bref
        self.compres = compres

     def evaluate(self, state):
         pres = state[0]
         X = self.compres * (pres - self.Pref)
         Bo = self.Bref / (1 + X + X * X / 2)

         return self.surf_oil_dens / Bo

class DensityWater(property_evaluator_iface):
    def __init__(self, pref = 1, Bref = 1.01, compres = 0, surface_wat_dens = 1025):
        super().__init__()
        self.surf_wat_dens = surface_wat_dens
        self.Pref = pref
        self.Bref = Bref
        self.compres = compres

    def evaluate(self, state):
        pres = state[0]
        X = self.compres * (pres - self.Pref)
        Bw = self.Bref / (1 + X + X * X / 2)

        return self.surf_wat_dens / Bw

class ViscosityOil(property_evaluator_iface):
    def __init__(self, pref=1, Viscoref=1, viscosibility=0):
        super().__init__()
        self.Pref = pref
        self.Viscoref = Viscoref
        self.viscosibility = viscosibility

    def evaluate(self, state):
        pres = state[0]
        Y = -self.viscosibility * (pres - self.Pref)
        muo = self.Viscoref / (1 + Y + Y * Y / 2)

        return muo

class ViscosityWater(property_evaluator_iface):
    def __init__(self, pref = 1, Viscoref = 0.31, viscosibility = 0):
        super().__init__()
        self.Pref = pref
        self.Viscoref = Viscoref
        self.viscosibility = viscosibility

    def evaluate(self, state):
        pres = state[0]
        Y = -self.viscosibility * (pres - self.Pref)
        muw = self.Viscoref / (1 + Y + Y * Y / 2)

        return muw

class Watersaturation(property_evaluator_iface):
    def __init__(self):
        super().__init__()

    def evaluate(self, state):
        wat_composition = state[1]
        water_density = DensityWater()
        wat_dens = water_density.evaluate(state)
        oil_density = DensityOil()
        oil_dens = oil_density.evaluate(state)
        water_sat = wat_composition * oil_dens / (wat_composition * oil_dens + wat_dens - wat_composition * wat_dens)
        water_sat = np.max([water_sat, 0])
        water_sat = np.min([water_sat, 1.0])

        return water_sat

class PhaseRelPerm(property_evaluator_iface):
    def __init__(self, phase, swc=0.15, sor=0.15, krwe=0.6, kroe=0.9, nw=2, no=2):
        super().__init__()
        self.phase = phase
        self.swc = swc
        self.sor = sor
        self.krwe = krwe
        self.kroe = kroe
        self.nw = nw
        self.no = no

    def evaluate(self, state):
        water_saturation = Watersaturation()
        wat_sat = water_saturation.evaluate(state)
        if self.phase == 'water':
            if wat_sat < self.swc:
                kr = 0
            elif wat_sat > 1 - self.sor:
                kr = self.krwe
            else:
                kr = self.krwe * ((wat_sat - self.swc) / (1 - self.sor - self.swc)) ** self.nw
        else:
            if 1 - wat_sat < self.sor:
                kr = 0
            elif 1 - wat_sat > 1 - self.sor:
                kr = self.kroe
            else:
                kr = self.kroe * ((1 - wat_sat - self.sor) / (1 - self.sor - self.swc)) ** self.no

        return kr

class Capillarypressure(property_evaluator_iface):
    def __init__(self, pentry=1e-6, lam=0.5, swc=0.2, sor=0.2):
        super().__init__()
        self.pd = pentry
        self.lam = lam
        self.swc = swc
        self.sor = sor

    def evaluate(self, state):
        water_saturation = Watersaturation()
        wat_sat = water_saturation.evaluate(state)
        eps = 1e-4

        pc = self.pd*(wat_sat-self.swc + eps)/(1 - self.swc - self.sor)

        return pc

class RockCompactionEvaluator(property_evaluator_iface):
    def __init__(self, pref=1, compres=4e-5):
        super().__init__()
        self.Pref = pref
        self.compres = compres

    def evaluate(self, state):
        pressure = state[0]

        return (1.0 + self.compres * (pressure - self.Pref))

