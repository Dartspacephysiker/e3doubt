"""
Here we simply take the outputs from parameterErrorEstimatesPrior run natively in R and compare them against what we get when we run parameterErrorEstimatesPrior via Python. All of the calls to the np.all function in the script should return "True"

SMH
2026/06/04
"""

import numpy as np
try:
    from rpy2.robjects.packages import importr
except:
    print("Couldn't import 'importr' from 'rpy2.robjects.packages'! Check rpy2 installation")
    importr = None
try:
    from rpy2 import robjects as robj
except:
    print("Couldn't import 'robjects' from rpy2! Check rpy2 installation")
    robj = None

ISgeometry = importr("ISgeometry")

SKI = (69.34,20.31)
KAR = (68.48,22.52)
KAI = (68.27,19.45)

to_floatv = lambda x: robj.vectors.FloatVector(x)
dropna = lambda x: x[~np.isnan(x)]

##############################
# Test 1

# define inputs
h,Ne,Ti,Te,nuin, fracOp, fwhmRange, resR, intT, NePr, TiPr, TePr, CollPr, ViPr,CompPr = 220,5e11,1000,1000,1,.9,1,5,30,1e9,1e5,1e5,0,1e5,1
runparms = dict(pm0 =to_floatv((30.5,16)),Tnoise=250,Pt=3.5e6,locTrans=to_floatv(SKI),locRec=(SKI,KAR,KAI))
runparms['locRec'] = robj.vectors.ListVector({f"rec{i+1}":to_floatv(rec) for i,rec in enumerate(runparms['locRec'])})

# correct answer 
correct0 = np.array((9.998562e+08, 5.209229e+01, 5.367131e+01, np.nan, 1.236332e+01, 8.825588e-02))
correct1 = np.array((9.993972e+08, 2.360629e+01, 2.482395e+01, np.nan, 5.574224e+00, 4.044672e-02))
correct2 = np.array((9.994398e+08, 2.452388e+01, 2.574195e+01, np.nan, 5.792638e+00, 4.197791e-02))                    

out = ISgeometry.parameterErrorEstimatesPrior(
    SKI[0], SKI[1], h, Ne, Ti, Te, nuin, fracOp, fwhmRange,
    resR,
    intT,
    NePr=NePr,
    TiPr=TiPr,
    TePr=TePr,
    CollPr=CollPr,
    ViPr=ViPr,
    CompPr=CompPr,
    **runparms)
los = out[0]

los0 = np.array(los[0])[~np.isnan(np.array(los[0]))]

print(np.all(np.isclose(dropna(correct0),dropna(np.array(los[0])))))
print(np.all(np.isclose(dropna(correct1),dropna(np.array(los[1])))))
print(np.all(np.isclose(dropna(correct2),dropna(np.array(los[2])))))


##############################
# Test 2

# define inputs
h, Ne, Ti, Te, nuin, fracOp, fwhmRange, resR, intT, NePr, TiPr, TePr, CollPr, ViPr, CompPr = 220,5e11,1000,1000,1,.5,1,5,30,1e12,1e5,1e5,0,1e5,1

runparms = dict(pm0 =to_floatv((30.5,16)),Tnoise=250,Pt=3.5e6,locTrans=to_floatv(SKI),locRec=(SKI,KAR,KAI),maxLag=100)
runparms['locRec'] = robj.vectors.ListVector({f"rec{i+1}":to_floatv(rec) for i,rec in enumerate(runparms['locRec'])})

# correct answer

correct0 = np.array((9.154733e+11, 2.081415e+03, 1.850955e+03, np.nan, 2.160395e+01, 9.995651e-01))
correct1 = np.array((7.172894e+11, 1.653838e+03, 1.522630e+03, np.nan, 9.740526e+00, 9.987713e-01))
correct2 = np.array((7.304602e+11, 1.681944e+03, 1.543843e+03, np.nan, 1.012219e+01, 9.988182e-01))
correctmult = np.array((5.775520e+11, 1.358710e+03, 1.304414e+03, np.nan, 1.149637e+02, 9.983265e-01))

out = ISgeometry.parameterErrorEstimatesPrior(
    SKI[0], SKI[1], h, Ne, Ti, Te, nuin, fracOp, fwhmRange,
    resR,
    intT,
    NePr=NePr,
    TiPr=TiPr,
    TePr=TePr,
    CollPr=CollPr,
    ViPr=ViPr,
    CompPr=CompPr,
    **runparms)
los = out[0]
multi = out[1]

print(np.all(np.isclose(dropna(correct0),dropna(np.array(los[0])))))
print(np.all(np.isclose(dropna(correct1),dropna(np.array(los[1])))))
print(np.all(np.isclose(dropna(correct2),dropna(np.array(los[2])))))
print(np.all(np.isclose(dropna(correctmult),dropna(np.array(multi)))))

##############################
# Test 3, same as above but with maxLag = 1000

#parameterErrorEstimatesPrior(lat=SKI[1],lon=SKI[2],alt=220,Ne=5e11,Ti=1000,Te=1000,Coll=1,Comp=.5,fwhmRange=1,resR=5,intTime=30,NePr=1e12,TiPr=1e5,TePr=1e5,CollPr=0,ViPr=1e5,CompPr=1,pm0=c(30.5,16),Tnoise=250,Pt=3.5e6,locTrans=SKI,locRec=list(SKI,KAR,KAI),maxLag=1000)

# define inputs
h, Ne, Ti, Te, nuin, fracOp, fwhmRange, resR, intT, NePr, TiPr, TePr, CollPr, ViPr, CompPr = 220,5e11,1000,1000,1,.5,1,5,30,1e12,1e5,1e5,0,1e5,1

runparms = dict(pm0 =to_floatv((30.5,16)),Tnoise=250,Pt=3.5e6,locTrans=to_floatv(SKI),locRec=(SKI,KAR,KAI),maxLag=1000)
runparms['locRec'] = robj.vectors.ListVector({f"rec{i+1}":to_floatv(rec) for i,rec in enumerate(runparms['locRec'])})

# correct answer
correct0 = np.array((1.220748e+10, 2.912122e+02, 2.848745e+02, np.nan, 1.099107e+01, 4.163825e-01))
correct1 = np.array((5.510368e+09, 1.412771e+02, 1.382804e+02, np.nan, 4.955520e+00, 2.022202e-01))
correct2 = np.array((5.726124e+09, 1.465767e+02, 1.434659e+02, np.nan, 5.149691e+00, 2.098013e-01))
correctmult = np.array((3.776968e+09, 9.787063e+01, 9.580192e+01, np.nan, 5.848813e+01, 1.401099e-01))

out = ISgeometry.parameterErrorEstimatesPrior(
    SKI[0], SKI[1], h, Ne, Ti, Te, nuin, fracOp, fwhmRange,
    resR,
    intT,
    NePr=NePr,
    TiPr=TiPr,
    TePr=TePr,
    CollPr=CollPr,
    ViPr=ViPr,
    CompPr=CompPr,
    **runparms)
los = out[0]
multi = out[1]

print(np.all(np.isclose(dropna(correct0),dropna(np.array(los[0])))))
print(np.all(np.isclose(dropna(correct1),dropna(np.array(los[1])))))
print(np.all(np.isclose(dropna(correct2),dropna(np.array(los[2])))))
print(np.all(np.isclose(dropna(correctmult),dropna(np.array(multi)))))



