#!/usr/bin/ipython3

import scipy as sp
from numpy.fft import fft,rfft,fftfreq,rfftfreq,fftshift
from scipy.interpolate import interp1d,CubicSpline
import sys
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

args = [0,'filename',1,0,-1,1e10,0.001,632.8e-9]#sys.argv # Arguments [filename,sampling[i.e. only correct for every nth data point],start index,end index,correction step,correction precision,wavelength]
#If it doesn't work properly try increasing the sampling to 2 or maybe 3.
#metadata1 = open(args[1],'r').readline()
null_point = 0#float(metadata1.split(',')[6]) # Your null point
null_p = null_point#float(metadata1.split(',')[6])*2e-3
sampling_d = 20e-9#float(metadata1.split(',')[5])*2e6 # sampling distance (speed/50 in meters)
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def remove_outliers(a):

    ret = a

    return ret

def bg(xs,a,b):
    return a*xs + b


wl = float(args[7])
def line(xs,phase):
    return sp.sin(2*sp.pi*xs/wl + phase)


samp = int(args[2])
#metadata2 = open(args[2],'r').readline()
fields = [('s1',float),('s2',float),('t1',int),('t2',int),('x',float)]
data = sp.loadtxt(args[1],skiprows = 1,dtype=fields)
#data2 = sp.loadtxt(args[2],skiprows = 1,dtype=fields)

#sampling_d2 = float(metadata2.split(',')[5])*2e-3
samp = int(args[2])
sort = sp.sort(data[100:-1000],order='x')
sort = sort[int(args[3]):int(args[4])]
sort['x'] *= 2e-3
bg_f,bg_c = curve_fit(bg, sort['x'], sort['s1'])

s = 100
sort['s1'] -= bg(sort['x'],*bg_f)
avg = 0 # Normalisation
print("Started normalisation")
for i in range(sort.size):
    if i < s:
        avg += abs(sort['s1'][i])
        continue
    elif i == s:
        avg +=abs(sort['s1'][i])
        avg = avg/(s+1)
        sort['s1'][0:s] *= 1.95/(sp.pi*avg)
        continue
    avg = (avg*s + abs(sort['s1'][i]))/(s+1)
    sort['s1'][i] *= 1.95/(sp.pi*avg)

tbd = []
for i in range(sort.size):
    if abs(sort['s1'][i]) > 1.0:

        tbd.append(i)

tbd = sp.array(tbd)
for d in tbd:
    sort = sp.delete(sort,d)
    tbd -= 1

n_axis = sort['x']-sort['x'][0]
null_p -= sort['x'][0]
phase_0 = sp.arcsin(sort['s1'][0])
if abs(line(n_axis[1],sp.arcsin(sort['s1'][0]))-sort['s1'][1]) > 0.01:
    phase_0 = sp.pi - phase_0

sep = float(args[5])
prec = float(args[6])
rnge = sp.arange(-3*wl*sep, 3*wl*sep,1)
rnge = [int(x) for x in rnge]
rnge = sp.array(rnge)
ref = dict([[x,line(x/sep,phase_0)] for x in rnge])
#ref = lambda x : ref_1per[x]
print("Correction starting:")
x = -1
corr = 0
for i in sp.array(range(int(sort.size/samp)))*samp:
    if abs(sort['s1'][i]) > 1:
        raise Error()
    printProgressBar(i, sort.size-samp,suffix = '%f5' % (sort['x'][i]*500))
    d = 0
    diff = n_axis[i] - x
    x = n_axis[i]
    if x - corr > wl:
        corr += wl
    while True:
        if abs(ref[int((d+x - corr)*sep)] - sort['s1'][i]) < prec:
            n_axis[i:]+=d
            if null_p <= x:
                null_p += d
            break
        if abs(ref[int((x-d - corr)*sep)] - sort['s1'][i]) < prec:
            n_axis[i:]-=d
            if null_p <= x:
                null_p -= d
            break
        d += 1/sep
    for f in range(samp-1):
        if i == 0:
            break
        n_axis[i - f - 1] = n_axis[i-samp] + (n_axis[i] - n_axis[i-samp])*(samp-f-1)/samp
'''
print('\n')
print("Secondary correction starting:")
x = -1
corr = 0
for i in sp.array(range(int(sort.size))):
    if abs(sort['s1'][i]) > 1:
        raise Error()
    printProgressBar(i, sort.size-1,suffix = '%f5' % sort['x'][i])
    d = 0
    diff = n_axis[i] - x
    x = n_axis[i]
    if x - corr > wl:
        corr += wl
    while True:
        if d > diff:
            break
        if abs(ref[int((d+x - corr)*sep)] - sort['s1'][i]) < prec:
            n_axis[i:]+=d
            if null_p <= x:
                null_p += d
            break
        if abs(ref[int((x-d - corr)*sep)] - sort['s1'][i]) < prec:
            n_axis[i:]-=d
            if null_p <= x:
                null_p -= d
            break
        d += 1/sep

#'''
n_axis -= null_p
#sort2 = sp.sort(data2[100:],order='x')
ndata_dtype = [('x',float),('s1',float),('s2',float)]
n_data = sp.array([(n_axis[i],sort['s1'][i],sort['s2'][i]) for i in range(sort.size)],dtype=ndata_dtype)
n_data = sp.sort(n_data,order='x')
print('\n')
print('Creating interpolator')
inter1 = interp1d(n_data['x'],n_data['s2'])
inter2 = interp1d(n_data['x'],n_data['s1'])

print('Creating interpolated data')
xs = sp.arange(n_axis[0],n_axis[-1]-2e-9,2e-9)
#xs2 = sp.arange(data2[100::samp]['x'].min(),data2[100::samp]['x'].max()-0.000001,0.000001)
ys = inter1(xs)
ys2 = inter2(xs)
'''
d1 = sampling_d*1e9
#d2 = sampling_d2
'''
d1 = 2
#d2 = 0.000000002
#'''
#'''
xf = fftfreq(xs.size,d=d1)
#xf2 = fftfreq(xs2.size,d=d2)
print('Taking FFT-S1')
yf = fft(ys)
print('Taking FFT-S2')
#yf = yf /(sp.exp(1j*2*sp.pi*null_point))
yf2 = fft(ys2)
'''
yf = fft(sort['s2'])
yf2 = fft(sort['s1'])
xf = fftfreq(sort['x'].size,d=d1)
#'''
#'''
xf = fftshift(xf)
#xf2 = fftshift(xf2)
yf = fftshift(yf)
yf2 = fftshift(yf2)
xx=xf[int(len(xf)/2+1):len(xf)]
#xx2=xf2[int(len(xf2)/2+1):len(xf2)]
yy=yf[int(len(xf)/2+1):len(xf)]
yy2=yf2[int(len(xf)/2+1):len(xf)]
'''
xx = xf
#xx2 = xf2
yy = yf
yy2 = yf2
#'''
#l =  yy*xx**2
#l2 = yy2*xx2**2

det_sens = sp.loadtxt('Detector_rel_sens.csv',dtype=[('x',float),('s',float)],delimiter=',',skiprows=1)
sens = interp1d(det_sens['x'],det_sens['s'])
def get_sens(x):
    try:
        return sens(x)/100
    except:
        return 1

for i in range(xx.size):
    yy[i] /= get_sens(1/xx[i])
    yy2[i] /= get_sens(1/xx[i])
    
plt.ion()
plt.figure('FFT')
plt.xlabel('Wavenumber [$cm^{-1}$]')
plt.plot(xf/100,sp.absolute(yf))
plt.plot(xf/100,sp.absolute(yf2))
plt.grid(True)
plt.figure('LAMBDA')
plt.xlabel('$\lambda$ [$nm$]')
plt.plot(1/xx,sp.absolute(yy))
plt.plot(1/xx,sp.absolute(yy2))
plt.grid(True)
plt.figure('SPECTRUM')
plt.xlabel('Position [mm]')
plt.plot(n_data['x']*500,n_data['s2'],'x-')
plt.plot(n_data['x']*500,n_data['s1'],'x-')
#plt.plot(n_data['x'],[abs(ref(int((n_data['x'][i]+null_p)*sep)) - n_data['s1'][i]) for i in range(sort.size)],'x-')
plt.plot(500*xs,line(xs, phase_0))
plt.grid(True)
plt.show()
