import pycte

chemModule = 'orchestra'  # phreeqc or orchestra
trsptModule = 'nativeTransport'  # nativeTransport, comsol, pflotran a faire
operatorSplitting = 'snia'  # 'additive', 'alternative', 'strang', 'symmetrical' ou 'snia'
# pblm additive ..

pycte.chemModule(chemModule)
pycte.trsptModule(trsptModule)
pycte.operatorSplitting(operatorSplitting)

pycte.maxTime(72000)
pycte.timeStep(720)
pycte.firstStepEquilibrium(True)

if chemModule == 'orchestra':
    systemSpeciation = ["Exch_X2-Ca", "Exch_X-K", "Exch_X-Na", "Ca+2", "CaOH+",
                         "Cl-", "H+", "H2", "K+", "Na+", "NaOH", "NO3-", "O2", "OH-"]
    primarySpeciesAq = ["Ca", "Cl", "K", "N", "Na"]

    speciesAttributes = {
        "con": [spc for spc in systemSpeciation if spc not in ["Exch_X2-Ca", "Exch_X-K", "Exch_X-Na"]],
        "solid": {"Exch_X2-Ca": 'Ca', "Exch_X-K": 'K', "Exch_X-Na": 'Na'},
    }

    transportedSpecies = ["Ca+2", "CaOH+", "Cl-", "H+", "H2", "K+", "Na+", "NaOH", "NO3-", "O2", "OH-"]
    
    pycte.inputVariableOrchestra([f'{spc}.tot' for spc in primarySpeciesAq ])
    pycte.outputVariableOrchestra([f'{spc}.con' if spc not in ["Exch_X2-Ca", "Exch_X-K", "Exch_X-Na"] else f'{spc}.solid' for spc in systemSpeciation ])

    
    pycte.systemSpeciation(systemSpeciation)
    pycte.primarySpeciesAq(primarySpeciesAq)
    # pycte.speciesAttributes(speciesAttributes)
    pycte.transportedSpecies(transportedSpecies)
    pycte.chemPath('chemistry1.inp')
    pycte.initialConditions("orchestraIC.txt")

elif chemModule == 'phreeqc':
    systemSpeciation = ["CaX2", "KX", "NaX", "NH4X", "Ca+2", "CaOH+", "Cl-", "H+", "H2", "K+",
                         "N2", "Na+", "NaOH", "NH3", "NH4+", "NO2-", "NO3-", "O2", "OH-"]
    pycte.systemSpeciation(systemSpeciation)
    pycte.initialConditions("IC.txt")
    pycte.chemPath("phreeqc.dat")

if trsptModule == 'comsol':
    pycte.trsptPath("comsol.mph")
elif trsptModule == 'nativeTransport':
    firstBoundary = {s: 0 for s in systemSpeciation}
    firstBoundary.update({'Ca+2': 0.6e-3, 'Cl-': 1.2e-3})

    pycte.dispersivity(0.002)
    pycte.velocity(0.002 / 720)
    pycte.ADE(True)
    pycte.firstBoundary(firstBoundary)
    pycte.boundaryConditions(['constant', 'closed'])

pycte.PIDnbr(1)
pycte.PIDextract(1)

# if __name__ == "__main__":
pycte.run()