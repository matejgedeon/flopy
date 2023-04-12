# from importlib import reload

import flopy
import numpy as np
import os
# from pathlib import Path
# from tempfile import TemporaryDirectory

#reload(flopy)

#%% creating modflow model

# temp_dir = TemporaryDirectory()
# workspace = Path(temp_dir.name)

workspace = "C:/Users/mgedeon/Desktop/temp/flopy-test"

mf = flopy.modflow.Modflow(modelname="test-flow",version="mf2005",
                           model_ws=workspace)

Lx = 1000.0
Ly = 1000.0
ztop = 0.0
zbot = -50.0
nlay = 1
nrow = 10
ncol = 10
delr = Lx / ncol
delc = Ly / nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)

dis = flopy.modflow.ModflowDis(
    mf, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm[1:]
)

ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
ibound[:, :, 0] = -1
ibound[:, :, -1] = -1
strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
strt[:, :, 0] = 10.0
strt[:, :, -1] = 0.0
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

lpf = flopy.modflow.ModflowLpf(mf, hk=10.0, vka=10.0, ipakcb=53)

rch = flopy.modflow.ModflowRch(mf)

lmt = flopy.modflow.ModflowLmt(mf)

spd = {(0, 0): ["print head", "print budget", "save head", "save budget"]}
oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

pcg = flopy.modflow.ModflowPcg(mf)

mf.write_input()

success, buff = mf.run_model()
assert success, "MODFLOW did not terminate normally."
#%% creating a MTHP model

mth = flopy.mthp.Mthp(modelname="test-mthp",modflowmodel=mf,model_ws=workspace,
                      verbose=True)
btn = flopy.mthp.MthBtn(mth,ncomp=1,mccomp=3,nper=3)
mcp = flopy.mthp.MthMcp(mth,component=['pH','Na',"Cl"])
#mcp1 = flopy.mthp.MthMcp.load(os.path.join(workspace,"test-mthp.mcp"),model=mth)
ssm = flopy.mthp.MthSsm(mth,crch={0:1.0,2:3},iSpec_rch={1:1001},
                        stress_period_data={0:[0,0,0,1.0,1,1001],
                                            1:[0,0,0,3.0,1,1002]})

# btn.write_file()
# mcp.write_file()
# ssm.write_file()
mth.write_input()
#%%
mth = None
mth = flopy.mthp.Mthp.load(os.path.join(workspace,"test-mthp.nam"),verbose=True,modflowmodel=mf,model_ws=workspace)

#%% loading a different model
workspace = "Y:\\2020 - Response\\22_MTHP\\MTHP-benchmarks\\MTHP-benchmarks\\simple2D"

mf = flopy.modflow.Modflow.load("simple2d.nam",verbose=True,
                                model_ws=os.path.join(workspace,"modflow"))
mth = flopy.mthp.Mthp.load("simple2d.nam",verbose=True,model_ws=workspace)
