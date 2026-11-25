"""
Regression tests of the PhreeqC / PhreeqcRM coupling: a column without transport (nativeTransport, no velocity, no
diffusion) speciated at every step must stay where the first speciation put it (fixed point), conserve the
elements, and run the kinetics as PhreeqC does on its own. Counter-charge species (speciationCharge) and
crossDependencies shared by the speciation and the transport.

    python -m unittest discover -s tests -v          (from the folder holding pycte ; PhreeqC: phreeqpy)
"""
import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import pycte

PHREEQC_DAT = Path(pycte.__file__).parent / "Examples" / "CationExchange" / "phreeqc.dat"
HAS_RM = importlib.util.find_spec("phreeqcrm") is not None


def run(columns, values, nSteps=3, **settings):
    """Runs pycte on 3 identical nodes without transport -> [commMtrx of steps 0 .. nSteps] (stepEnd outputs)."""
    with tempfile.TemporaryDirectory() as folder:
        ic = Path(folder) / "ic.txt"
        rows = [[x] + list(values) for x in (0.05, 0.15, 0.25)]
        ic.write_text("%x\t" + "\t".join(columns) + "\n" + "\n".join("\t".join(repr(float(v)) for v in row)
                                                                      for row in rows) + "\n")
        config = dict(chemistryModule="phreeqc", transportModule="nativeTransport", chemPath=str(PHREEQC_DAT),
                      initialConditions=str(ic), coupledVariables=list(columns), timeStep=600,
                      maxTime=600 * nSteps, firstStepEquilibrium=True, ADE=True, velocity=0, dispersivity=0,
                      diffCoeff=0, boundaryConditions=["closed", "closed"],
                      output={"stepEnd": list(range(nSteps + 1))})
        config.update(settings)
        cwd = os.getcwd()
        os.chdir(folder)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pycte.run(config)
            return [pd.read_csv(Path(folder) / "stepEnd" / f"StepEnd_{k}.txt", sep="\t") for k in range(nSteps + 1)]
        finally:
            os.chdir(cwd)


class FixedPoint(unittest.TestCase):

    def assertSameState(self, states, columns, rtol=1e-7, atol=1e-12):
        for k, state in enumerate(states[1:], 1):
            for c in columns:
                np.testing.assert_allclose(state[c], states[0][c], rtol=rtol, atol=atol,
                                           err_msg=f"step {k}, {c} moved away from the speciated state of step 0")

    # every species of the system is coupled (a species left out is lost at every step)
    acid = (["chargebalance", "pH", "pe", "H2O", "H+", "OH-", "H2", "O2", "Na+", "NaOH", "Cl-"],
            [0.0, 3.0, 4.0, 55.50868, 1e-3, 1e-11, 0.0, 0.0, 1e-4, 0.0, 1.1e-3])

    def test_solMod_keeps_water_and_free_H(self):
        """solMod: H+ counted in -total_h (phreeqc.dat calls its state H(1)), no water oxidised to O2. pe left out:
        without a redox couple it only follows traces of O2 and H2."""
        columns, values = self.acid
        states = run(columns, values, solMod=True, crossDependencies={"speciation": ["chargebalance", "pH", "pe"]})
        self.assertSameState(states, [c for c in columns if c != "pe"])

    @unittest.skipUnless(HAS_RM, "phreeqcrm not installed")
    def test_solMod_keeps_water_and_free_H_PhreeqcRM(self):
        columns, values = self.acid
        states = run(columns, values, chemistryModule="phreeqcrm", solMod=True,
                     crossDependencies={"speciation": ["chargebalance", "pH", "pe"]})
        self.assertSameState(states, [c for c in columns if c != "pe"])

    surface = (["chargebalance", "pH", "pe", "H2O", "H+", "OH-", "H2", "O2", "Na+", "NaOH", "Cl-", "Ca+2", "CaOH+",
                "Hfo_wOH", "Hfo_wOH2+", "Hfo_wO-", "Hfo_wOCa+"],
               [0.0, 8.0, 4.0, 55.50868, 1e-8, 1e-6, 0.0, 0.0, 1e-3, 0.0, 3e-3, 1e-3, 0.0, 1e-2, 0.0, 0.0, 0.0])

    def test_solMod_surface_charge(self):
        """solMod with a charged surface: the charge of the surface species goes back to the solution (-cb)."""
        columns, values = self.surface
        states = run(columns, values, solMod=True, crossDependencies={"speciation": ["chargebalance", "pH", "pe"]})
        self.assertSameState(states, columns)

    @unittest.skipUnless(HAS_RM, "phreeqcrm not installed")
    def test_solMod_surface_charge_PhreeqcRM(self):
        columns, values = self.surface
        states = run(columns, values, chemistryModule="phreeqcrm", solMod=True,
                     crossDependencies={"speciation": ["chargebalance", "pH", "pe"]})
        self.assertSameState(states, columns)

    def test_surface_without_charge_balance(self):
        """Charge balance not coupled: the solution is rebuilt neutral (the system is), the surface charge is not
        given to it again."""
        columns, values = self.surface[0][1:], self.surface[1][1:]
        for chemistryModule in ("phreeqc",) + (("phreeqcrm",) if HAS_RM else ()):
            with self.subTest(chemistryModule=chemistryModule):
                states = run(columns, values, chemistryModule=chemistryModule, solMod=True,
                             crossDependencies={"speciation": ["pH", "pe"]})
                self.assertSameState(states, columns)


class PreliminarEquilibrium(unittest.TestCase):
    """preliminarEquilibrium from a state that is not at equilibrium (supersaturated with calcite): without
    transport, the same states as without preliminarEquilibrium (calcite precipitates once, Ca is conserved)."""

    columns = ["chargebalance", "pH", "pe", "H2O", "H+", "OH-", "H2", "O2", "Ca+2", "CaOH+", "CO3-2", "HCO3-", "CO2",
               "CH4", "CaCO3", "CaHCO3+", "Na+", "NaOH", "NaCO3-", "Cl-", "Calcite"]
    values = [0.0, 8.0, 4.0, 55.50868, 1e-8, 1e-6, 0.0, 0.0, 5e-3, 0.0, 1e-4, 9.8e-3, 0.0, 0.0, 0.0, 0.0, 1.0099e-4,
              0.0, 0.0, 1e-4, 1e-3]          # electroneutral (chargebalance 0)

    def check(self, **settings):
        common = dict(firstStepEquilibrium=False, crossDependencies={"speciation": ["chargebalance", "pH", "pe"]},
                      **settings)
        preliminar = run(self.columns, self.values, preliminarEquilibrium=True, **common)
        reference = run(self.columns, self.values, **common)
        for k, (state, expected) in enumerate(zip(preliminar[1:], reference[1:]), 1):
            for c in ("Ca+2", "CaCO3", "CaHCO3+", "Calcite", "pH", "H2O"):
                np.testing.assert_allclose(state[c], expected[c], rtol=1e-7, atol=1e-14,
                                           err_msg=f"step {k}, {c} : preliminarEquilibrium without transport differs")

    def test_solMod(self):
        self.check(solMod=True)

    def test_solution(self):
        self.check(solMod=False)


class Kinetics(unittest.TestCase):
    """Calcite dissolution with the rate of phreeqc.dat (surface area ~ (M/M0)^0.6), 4 steps of 600 s."""

    kinetics = "Calcite\n-parms 5 0.6\n-tol 1e-10"
    columns = ["pH", "pe", "H+", "OH-", "Ca+2", "CO3-2", "HCO3-", "CO2", "CaCO3", "CaHCO3+", "Cl-", "Calcite"]
    values = [3.0, 4.0, 1e-3, 1e-11, 1e-5, 0.0, 1e-5, 1e-5, 0.0, 0.0, 1.02e-3, 2e-4]

    def reference(self, nSteps=4, dt=600):
        """PhreeqC alone: same solution, KINETICS over nSteps x dt."""
        import phreeqpy.iphreeqc.phreeqc_dll as phreeqcDll
        solver = phreeqcDll.IPhreeqc()
        try:
            solver.load_database(str(PHREEQC_DAT))
            solver.run_string("SOLUTION 1\n-units mol/kgw\npH 3\nCa 1e-5\nC(4) 2e-5\nCl 1.02e-3\nEND\n"
                              f"KINETICS 1\n{self.kinetics}\n-m0 2e-4\n-steps {nSteps * dt} in {nSteps} steps\n"
                              "INCREMENTAL_REACTIONS true\nUSE solution 1\n"
                              "SELECTED_OUTPUT\n-reset false\n-kinetic_reactants Calcite\nEND\n")
            out = solver.get_selected_output_array()
        finally:
            solver.destroy_iphreeqc()
        return np.array([row[0] for row in out[1:]])

    def coupled(self, timeUnit, timeStep):
        states = run(self.columns, self.values, nSteps=4, timeUnit=timeUnit, timeStep=timeStep,
                     maxTime=4 * timeStep, kinetics=self.kinetics, kineticsSpecies=["Calcite"],
                     crossDependencies={"speciation": ["pH", "pe"]})
        return np.array([state["Calcite"].iloc[0] for state in states[1:]])

    def test_rate_keeps_M0(self):
        np.testing.assert_allclose(self.coupled("s", 600), self.reference(), rtol=1e-4)

    def test_time_unit_a(self):
        """timeUnit 'a' (year): the kinetics step is given to PhreeqC in seconds."""
        np.testing.assert_allclose(self.coupled("a", 600 / 31557600), self.coupled("s", 600), rtol=1e-6)


class ChargeBalance(unittest.TestCase):
    """speciationCharge: the counter-charge species takes the charge imbalance (NaCl with 2e-4 eq of excess Cl)."""

    columns = ["chargebalance", "pH", "pe", "H+", "OH-", "Na+", "Cl-"]
    values = [0.0, 7.0, 4.0, 1e-7, 1e-7, 1e-3, 1.2e-3]
    cross = {"speciation": ["chargebalance", "pH", "pe"]}

    def test_pH(self):
        """default speciesCharge ['pH']: 'pH 7 charge' on one line (a lone 'charge' line stopped PhreeqC)"""
        states = run(self.columns, self.values, nSteps=1, speciationCharge=True, crossDependencies=self.cross)
        np.testing.assert_allclose(states[1]["chargebalance"], 0.0, atol=1e-10)
        self.assertLess(states[1]["pH"].iloc[0], 4.0)

    def test_species_name(self):
        """a species name (Cl-) carries the charge balance through its master species (Cl)"""
        states = run(self.columns, self.values, nSteps=1, speciationCharge=True, speciesCharge=["Cl-"],
                     crossDependencies=self.cross)
        np.testing.assert_allclose(states[1]["chargebalance"], 0.0, atol=1e-10)
        np.testing.assert_allclose(states[1]["Cl-"], 1e-3, rtol=1e-3)

    def test_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "Xyz"):
            run(self.columns, self.values, nSteps=1, speciationCharge=True, speciesCharge=["Xyz"],
                crossDependencies=self.cross)


class CrossDependencies(unittest.TestCase):

    def test_speciation_and_transport(self):
        """a temperature coupled with both the speciation and the transport is still read by PhreeqC"""
        columns = ["tk", "chargebalance", "pH", "pe", "H+", "OH-", "Na+", "Cl-"]
        values = [308.15, 0.0, 7.0, 4.0, 1e-7, 1e-7, 1e-3, 1e-3]
        states = run(columns, values, nSteps=2,
                     crossDependencies={"speciation": ["tk", "chargebalance", "pH", "pe"], "transport": ["tk"]})
        np.testing.assert_allclose(states[2]["tk"], 308.15)


class TemperatureInKelvin(unittest.TestCase):
    """Every input temperature is in K: tempDefault, chemistry['temperature'] (nativeSpeciation), the tk column; the
    crossDependency tc (temperature in °C) is refused."""

    columns = ["chargebalance", "pH", "pe", "H+", "OH-", "Na+", "Cl-"]
    values = [0.0, 7.0, 4.0, 1e-7, 1e-7, 1e-3, 1e-3]
    cross = {"speciation": ["chargebalance", "pH", "pe"]}

    def test_tempDefault_in_K(self):
        """tempDefault 308.15 K: same speciation as a tk column at 308.15 K, other than at the default 298.15 K"""
        default = run(self.columns, self.values, nSteps=1, crossDependencies=self.cross)
        warm = run(self.columns, self.values, nSteps=1, crossDependencies=self.cross, tempDefault=308.15)
        column = run(["tk"] + self.columns, [308.15] + self.values, nSteps=1,
                     crossDependencies={"speciation": ["tk"] + self.cross["speciation"]})
        explicit = run(self.columns, self.values, nSteps=1, crossDependencies=self.cross, tempDefault=298.15)
        np.testing.assert_allclose(warm[1]["OH-"], column[1]["OH-"], rtol=1e-8)
        np.testing.assert_allclose(explicit[1]["OH-"], default[1]["OH-"], rtol=1e-8)
        # pH imposed at 7: OH- follows Kw, about 2.1 times larger at 35 °C than at 25 °C
        ratio = (warm[1]["OH-"] / default[1]["OH-"]).to_numpy()
        np.testing.assert_allclose(ratio, 2.1, rtol=0.1)

    def test_tempDefault_in_C_refused(self):
        with self.assertRaisesRegex(ValueError, "tempDefault.*K"):
            run(self.columns, self.values, nSteps=1, crossDependencies=self.cross, tempDefault=25)

    def test_tc_refused(self):
        for kind, word in (("speciation", "tc"), ("speciation", "temperature(C)"), ("transport", "tempC")):
            with self.subTest(word=word), self.assertRaisesRegex(ValueError, r"°C is no longer coupled"):
                run(["tc"] + self.columns, [25.0] + self.values, nSteps=1,
                    crossDependencies={**self.cross, kind: self.cross["speciation"] * (kind == "speciation") + [word]})

    def test_tk_column_in_C_refused(self):
        with self.assertRaisesRegex(ValueError, "temperature in K"):
            run(["tk"] + self.columns, [25.0] + self.values, nSteps=1,
                crossDependencies={"speciation": ["tk"] + self.cross["speciation"]})

    def test_xgems_tempDefault_in_K(self):
        from pycte.Source import gems
        commMtrx = pd.DataFrame({"tk": [300.0, 310.0]})
        temperature, _ = gems.nodeConditions({"tempDefault": 308.15}, commMtrx, {})
        np.testing.assert_allclose(temperature, 308.15)
        temperature, _ = gems.nodeConditions({}, commMtrx, {})
        np.testing.assert_allclose(temperature, 298.15)
        temperature, _ = gems.nodeConditions({}, commMtrx, {"tk": "temperature(K)"})
        np.testing.assert_allclose(temperature, [300.0, 310.0])

    def test_native_chemistry_temperature_in_K(self):
        from pycte.Source import chemistryDataframe, keywordValidation, nativeSpeciation
        db = chemistryDataframe._load_database(str(PHREEQC_DAT))
        cold = nativeSpeciation.ChemicalSystem(db, elements=["Na", "Cl"])
        warm = nativeSpeciation.ChemicalSystem(db, elements=["Na", "Cl"], temperature=308.15)
        self.assertEqual(cold.tk, 298.15)
        self.assertEqual(warm.tk, 308.15)
        self.assertEqual(chemistryDataframe.Config(elements=("Na",)).temperature, 298.15)
        self.assertEqual(keywordValidation.errorsOf("chemistry", {"elements": ["Na"], "temperature": 308.15}), [])
        self.assertEqual(keywordValidation.errorsOf("chemistry", {"elements": ["Na"], "temperature": "tk"}), [])
        self.assertTrue(keywordValidation.errorsOf("chemistry", {"elements": ["Na"], "temperature": 25}))

    def test_native_run_temperature_in_K(self):
        """nativeSpeciation run, chemistry['temperature'] in K: neutral NaCl at pKw / 2, 7.00 at 25 °C, 6.84 at 35 °C"""
        pH = {}
        for T in (298.15, 308.15):
            states = run(["H", "O", "Na", "Cl", "pH"], [111.01736, 55.50868, 1e-3, 1e-3, 7.0], nSteps=1,
                         chemistryModule="nativeSpeciation", chemPath=None,
                         chemistry={"elements": ["Na", "Cl"], "chemPath": str(PHREEQC_DAT), "temperature": T,
                                    "exchanger": None, "diagnostics": {"pH": "pH"}})
            pH[T] = states[1]["pH"].to_numpy()
        np.testing.assert_allclose(pH[298.15], 7.00, atol=0.005)
        np.testing.assert_allclose(pH[308.15], 6.84, atol=0.005)


if __name__ == "__main__":
    unittest.main()
