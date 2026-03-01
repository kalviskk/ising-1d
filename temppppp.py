# generate_he_inputs.py
# This script generates GAMESS input files for He: RHF ground state + CIS singles

# -----------------------------
# 1. RHF ground state input
# -----------------------------
rhf_input = """ $CONTRL SCFTYP=RHF RUNTYP=ENERGY MULT=1 MAXIT=50 ISPHER=1 $END
$SYSTEM TIMLIM=525600 MEMORY=2000000 $END
$BASIS GBASIS=ACC6C DIFFSP=.TRUE. DIFFS=.TRUE. $END
$GUESS GUESS=HCORE $END
$DATA
He atom
C1
He  2.0  0.0 0.0 0.0
$END
"""

with open("he_rhf.inp", "w") as f:
    f.write(rhf_input)

print("Generated he_rhf.inp (RHF ground state)")

# -----------------------------
# 2. CIS singles input
# -----------------------------
cis_input = """ $CONTRL SCFTYP=RHF RUNTYP=CI CITYP=CIS MULT=1 MAXIT=50 $END
$SYSTEM TIMLIM=525600 MEMORY=2000000 $END
$BASIS GBASIS=ACC6C DIFFSP=.TRUE. DIFFS=.TRUE. $END
$GUESS GUESS=MOREAD $END
$CIS NSTATE=5 $END
$DATA
He atom
C1
He  2.0  0.0 0.0 0.0
$END
"""

with open("he_cis.inp", "w") as f:
    f.write(cis_input)

print("Generated he_cis.inp (CIS singles using RHF orbitals)")
