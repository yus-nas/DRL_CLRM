from darts.engines import *
from correlation.properties_correlations import *

# Define operator evaluator class
class dead_oil_acc_flux_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()
        self.property = property_container

    def evaluate(self, state, values):
        oil_dens = self.property.density_ev['oil'].evaluate(state)
        wat_dens = self.property.density_ev['water'].evaluate(state)
        oil_visco = self.property.viscosity_ev['oil'].evaluate(state)
        wat_visco = self.property.viscosity_ev['water'].evaluate(state)
        wat_sat = self.property.watersat_ev.evaluate(state)
        oil_relp = self.property.rel_perm_ev['oil'].evaluate(state)
        wat_relp = self.property.rel_perm_ev['water'].evaluate(state)
        rock_cp = self.property.rock_compress_ev.evaluate(state)

        # acc part
        values[0] = rock_cp * wat_sat * wat_dens
        values[1] = rock_cp * (1 - wat_sat) * oil_dens

        # flux operator
        values[2] = wat_dens * (wat_relp / wat_visco)   # water component in water phase
        values[3] = oil_dens * (oil_relp / oil_visco)  # oil component in oil phase

        return 0

class dead_oil_acc_flux_capillary_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()
        self.property = property_container

    def evaluate(self, state, values):
        oil_dens = self.property.density_ev['oil'].evaluate(state)
        wat_dens = self.property.density_ev['water'].evaluate(state)
        oil_visco = self.property.viscosity_ev['oil'].evaluate(state)
        wat_visco = self.property.viscosity_ev['water'].evaluate(state)
        wat_sat = self.property.watersat_ev.evaluate(state)
        oil_relp = self.property.rel_perm_ev['oil'].evaluate(state)
        wat_relp = self.property.rel_perm_ev['water'].evaluate(state)
        rock_cp = self.property.rock_compress_ev.evaluate(state)
        pc = self.property.capillary_ev.evaluate(state)

        # acc part
        values[0] = rock_cp * wat_sat * wat_dens
        values[1] = rock_cp * (1 - wat_sat) * oil_dens

        # flux operator
        # (1) water phase
        values[2] = wat_dens             # water density operator
        values[3] = 0                    # reference phase, pc = 0
        values[4] = wat_dens * (wat_relp / wat_visco)   # water component in water phase
        values[5] = 0                                   # water component in oil phase

        # (2) oil phase
        values[6] = oil_dens             # oil density operator
        values[7] = 0                    # pc, should be -pc, here we ignore pc
        values[8] = 0                                  # oil component in water phase
        values[9] = oil_dens * (oil_relp / oil_visco)  # oil component in oil phase

        return 0

class dead_oil_rate_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()
        self.property = property_container

    def evaluate(self, state, values):
        oil_dens = self.property.density_ev['oil'].evaluate(state)
        wat_dens = self.property.density_ev['water'].evaluate(state)
        oil_visco = self.property.viscosity_ev['oil'].evaluate(state)
        wat_visco = self.property.viscosity_ev['water'].evaluate(state)
        oil_relp = self.property.rel_perm_ev['oil'].evaluate(state)
        wat_relp = self.property.rel_perm_ev['water'].evaluate(state)

        # surface density
        self.surface_wat_dens = self.property.density_ev['water'].surf_wat_dens
        self.surface_oil_dens = self.property.density_ev['oil'].surf_oil_dens

        # flux in reservoir condition
        wat_flux = wat_dens * (wat_relp / wat_visco)
        oil_flux = oil_dens * (oil_relp / oil_visco)

        # convert to surface condition
        values[0] = wat_flux / self.surface_wat_dens
        values[1] = oil_flux/ self.surface_oil_dens
        values[2] = values[0] + values[1]

        return 0


class Saturation():
    def __init__(self, property_container):
        super().__init__()
        self.property = property_container

    def evaluate(self, state):
        oil_dens = self.property.density_ev['oil'].evaluate(state)
        wat_dens = self.property.density_ev['water'].evaluate(state)
        oil_visco = self.property.viscosity_ev['oil'].evaluate(state)
        wat_visco = self.property.viscosity_ev['water'].evaluate(state)
        wat_sat = self.property.watersat_ev.evaluate(state)
        oil_relp = self.property.rel_perm_ev['oil'].evaluate(state)
        wat_relp = self.property.rel_perm_ev['water'].evaluate(state)
        rock_cp = self.property.rock_compress_ev.evaluate(state)

        return wat_sat, oil_relp, wat_relp

class dead_oil_acc_flux_well_evaluator_python(operator_set_evaluator_iface):
    def __init__(self, property_container):
        super().__init__()
        self.property = property_container

    def evaluate(self, state, values):
        oil_dens = self.property.density_ev['oil'].evaluate(state)
        wat_dens = self.property.density_ev['water'].evaluate(state)
        oil_visco = self.property.viscosity_ev['oil'].evaluate(state)
        wat_visco = self.property.viscosity_ev['water'].evaluate(state)
        wat_sat = self.property.watersat_ev.evaluate(state)
        oil_relp = self.property.rel_perm_ev['oil'].evaluate(state)
        wat_relp = self.property.rel_perm_ev['water'].evaluate(state)
        rock_cp = self.property.rock_compress_ev.evaluate(state)

        # acc part
        values[0] = rock_cp * wat_sat * wat_dens
        values[1] = rock_cp * (1 - wat_sat) * oil_dens

        # flux operator
        values[2] = wat_dens * (wat_relp / wat_visco)   # water component in water phase
        values[3] = oil_dens * (oil_relp / oil_visco)  # oil component in oil phase

        return 0