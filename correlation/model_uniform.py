from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from correlation.dead_oil_uniform import DeadOil
from darts.engines import value_vector
import numpy as np
from correlation.properties_correlations import *
from darts.tools.keyword_file_tools import load_single_keyword

class Model(DartsModel):

    def __init__(self):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        """Reservoir construction"""
        # reservoir geometry： for realistic case, one just needs to load the data and input it
        self.reservoir = StructReservoir(self.timer, nx=60, ny=220, nz=1, dx=6.0, dy=3.0, dz=0.6, permx=300, permy=300,
                                         permz=300, poro=0.2, depth=100)
        # well model or boundary conditions
        well_dia = 0.30
        well_rad = well_dia / 2
        self.reservoir.add_well("I1", wellbore_diameter=well_dia)
        self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=30, j=110, k=1, well_radius=well_rad,
                                       multi_segment=False)

        self.reservoir.add_well("P1", wellbore_diameter=well_dia)
        self.reservoir.add_perforation(self.reservoir.wells[-1], 1, 1, 1, well_radius=well_rad, multi_segment=False)

        self.reservoir.add_well("P2", wellbore_diameter=well_dia)
        self.reservoir.add_perforation(self.reservoir.wells[-1], 60, 1, 1, well_radius=well_rad, multi_segment=False)

        self.reservoir.add_well("P3", wellbore_diameter=well_dia)
        self.reservoir.add_perforation(self.reservoir.wells[-1], 1, 220, 1, well_radius=well_rad, multi_segment=False)

        self.reservoir.add_well("P4", wellbore_diameter=well_dia)
        self.reservoir.add_perforation(self.reservoir.wells[-1], 60, 220, 1, well_radius=well_rad, multi_segment=False)

        """Physical properties"""
        self.property_container = PropertyContainer(phase_name=['water', 'oil'], component_name=['water', 'oil'],
                                                    rate_name=['water', 'oil', 'liquid'])

        # Define property evaluators based on custom properties
        self.property_container.density_ev = dict([('water', DensityWater()), ('oil', DensityOil())])
        self.property_container.viscosity_ev = dict([('water', ViscosityWater()), ('oil', ViscosityOil())])
        self.property_container.watersat_ev = Watersaturation()
        self.property_container.rel_perm_ev = dict([('water', PhaseRelPerm("water")), ('oil', PhaseRelPerm("oil"))])
        self.property_container.capillary_ev = Capillarypressure()
        self.property_container.rock_compress_ev = RockCompactionEvaluator()

        # create physics
        self.grav = 0
        self.physics = DeadOil(self.timer, n_points=400, min_p=0, max_p=1000, min_z=1e-13,
                               property_container=self.property_container, grav=self.grav)

        self.params.first_ts = 0.01
        self.params.mult_ts = 2
        self.params.max_ts = 25
        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-6
        # self.params.newton_type = 2
        # self.params.newton_params = value_vector([0.2])

        self.runtime = 1000
        self.inj = value_vector([0.999])

        self.timer.node["initialization"].stop()

    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=400, uniform_composition=[0.2357])

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_water_inj(100, self.inj)
                w.constraint = self.physics.new_bhp_water_inj(450, self.inj)
            else:
                w.control = self.physics.new_rate_liq_prod(20)
