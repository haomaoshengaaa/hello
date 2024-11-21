https://github.com/QijingZheng/VaspBandUnfolding/tree/master



This code on github is for extracting planewave coefficients from WAVECAR. You can follow the instructions on this github page to intall it and then extract the planewave coefficients like this:

from vaspwfc import vaspwfc

pswfc = vaspwfc("./WAVECAR")     # load the WAVECAR file in current folder



# extract the planewave coefficients for the first spin, the second k point (corresponding to the second k point in EIGENVAL file which I already sent you earlier), and the seventh band (also in EIGENVAL file).Note that ispin,ikpt,iband all start  from 1. Given that our calculation is non-spin-polarised, ispin can only be strictly set to 1.  And you can choose any ikpt and iband you want (you could easily know how many k points and bands there are in EIGENVAL file)

coeff = pswfc.readBandCoeff(ispin=1, ikpt=2, iband=7)

# To get the G vectors corresponding to the planewave coefficients (each G vector has its own single planewave coefficient), you can call this method. The obtained gvectors are in the form of direct coordinates in reciprocal space.
ikpt = 2
gvec = pswfc.gvectors(ikpt)
    
There is an internal method in this code called "wfc_r". If you want to know how it constructs the realspace peuso-wavefunction (the wave function which does not include the core part), you may have to read the content in this method (click the link below and this method starts around line 486)

https://github.com/QijingZheng/VaspBandUnfolding/blob/master/vaspwfc.py
