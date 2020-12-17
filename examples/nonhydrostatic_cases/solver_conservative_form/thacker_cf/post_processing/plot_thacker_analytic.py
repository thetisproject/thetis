# Analytic solution to Thacker wetting-drying test case

import numpy as np
from matplotlib import pyplot as plt
import csv

plt.rc('font', family='serif', size=24)

filename1 = '/data/wp715/firedrake/src/thetis/cases/thacker/18h.csv'
filename2 = '/data/wp715/firedrake/src/thetis/cases/thacker/24h.csv'


def read_line(filename):
    fileReader = csv.reader(open(filename, 'rb'), delimiter='\t')
    dCoords = []
    eleVals = []
    for row in fileReader:
        eleVals.append(float(row[1]))
        dCoords.append(float(row[0])/1000.)
    return dCoords, eleVals


dplot18, eleplot18 = read_line(filename1)
dplot24, eleplot24 = read_line(filename2)

# Parameters
D0 = 50.0
L = 430620
eta0 = 2.0
A = ((D0+eta0)**2-D0**2)/((D0+eta0)**2+D0**2)
R = 430620.0
omega = (8*9.81*D0/(R**2))**0.5

xplot = np.linspace(-475823.23, 475823.23, 1000)

# Bathymetry and analytic solution
bathy = -D0*(1-(xplot**2)/(L**2))


def analytic(t):
    return D0*(((1-A**2)**0.5)/(1-A*np.cos(omega*t)) - 1 - (xplot**2)/(R**2)*((1-A**2)/((1-A*np.cos(omega*t))**2)-1))


# Plot solution
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)

plt.plot(xplot/1000, bathy, 'k', linewidth=4)
plt.plot(xplot/1000, analytic(0), 'k', linewidth=2)
# plt.plot(xplot/1000, analytic(6*3600), 'r', linewidth=2)
# plt.plot(xplot/1000, analytic(12*3600), 'g', linewidth=2)
plt.plot(xplot/1000, analytic(18*3600), 'y', linewidth=2)
plt.plot(dplot18, eleplot18, 'go', linewidth=1)
plt.plot(xplot/1000, analytic(24*3600), 'b', linewidth=2)
plt.plot(dplot24, eleplot24, 'r+', linewidth=8)

plt.xlim(-500, 500)
plt.ylim(-8, 12)
plt.xlabel('r / km')
plt.ylabel('elevation / m')
plt.tight_layout()
plt.show()
