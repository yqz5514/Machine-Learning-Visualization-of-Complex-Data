#%%
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linalg as LA
import numpy as np
from tabulate import tabulate
from scipy.optimize import curve_fit
plt.style.use('seaborn-whitegrid')

#%%
x = np.linspace(-10,10,1000)
x = np.array(x, dtype=np.complex)
y1 = x ** 2
y2 = x ** 3
y3 = x ** (1/2)
y4 = x ** (1/3)

fig, ax = plt.subplots(2,2)

ax[0,0].plot(x,y1)
ax[0,0].set_title('$f(x)=x^{2}$')
ax[0,0].set_xlabel('x')
ax[0,0].set_ylabel('y')

ax[0,1].plot(x,y2)
ax[0,1].set_title('$f(x)=x^{3}$')
ax[0,1].set_xlabel('x')
ax[0,1].set_ylabel('y')

ax[1,0].plot(x,y3)
ax[1,0].set_title('$f(x)=\sqrt{x}$')
ax[1,0].set_xlabel('x')
ax[1,0].set_ylabel('y')

ax[1,1].plot(x,y4)
ax[1,1].set_title('$f(x)=\sqrt[3]{x}$')
ax[1,1].set_xlabel('x')
ax[1,1].set_ylabel('y')

plt.tight_layout()
plt.show()

#%%
x = np.linspace(1,10)
# y = [10 ** el for el in x]
# z = [2 ** el for el in x]
#
# fig = plt.figure(figsize=(10,8))
# ax1 = fig.add_subplot(2,2,1)
# ax1.plot(x,y, color = 'blue')
# ax1.set_yscale('log')
ax1.set_title('Logarithmic plot of $10^{x}$')
# ax1.set_ylabel('$y = 10^{x} $')
# plt.grid(visible=True, which='both')
#
#
# ax2 = fig.add_subplot(2,2,2)
# ax2.plot(x,y, color = 'blue')
# ax2.set_yscale('linear')
# ax2.set_title('linear plot of $10^{x}$')
# ax2.set_ylabel('$y = 10^{x} $')
# plt.grid(visible=True, which='both')
#
#
# ax3 = fig.add_subplot(2,2,3)
# ax3.plot(x,z, color = 'green')
# ax3.set_yscale('log')
# ax3.set_title('Logarithmic plot of $10^{x}$')
# ax3.set_ylabel('$y = 2^{x} $')
# plt.grid(visible=True, which='both')
#
#
# ax4 = fig.add_subplot(2,2,4)
# ax4.plot(x,z, color = 'magenta')
# ax4.set_yscale('linear')
# ax4.set_title('Linear plot of $2^{x}$')
# ax4.set_ylabel('$y = 2^{x} $')
# plt.grid(visible=True, which='both')
#
# plt.tight_layout()
# plt.show()
# x = np.linspace(0, 2*np.pi, 1000)
# y1 = np.sin(x)
# y2 = np.cos(x)
#
# font1 = {'family':'serif','color':'blue', 'size':20}
# font2 = {'family':'serif','color':'darkred', 'size':15}
#
#
# c1 = np.random.rand(1,3)
# c2 = np.random.rand(1,3)
#
# plt.figure(figsize=(12,8))
# plt.subplot(2,1,1)
# plt.plot(x, y1, label = r'$\sin(x)$', lw = 4, color = c1, linestyle = ':')
# plt.plot(x, y2, label = r'$\cos(x)$', lw = 2.5, color = c2, linestyle = '-')
# plt.legend(loc = 'lower left')
# plt.title('sine and cosine graph', fontdict=font1)
# plt.xlabel('time', fontdict=font2)
# plt.ylabel('Mag', fontdict=font2)
# plt.grid(axis='x')
#
# plt.subplot(2,1,2)
# plt.plot(x, y1+y2, label = r'$\sin(x)+\cos(x)$', lw = 4, color = c1, linestyle = '-.')
# # plt.plot(x, y2, label = r'$\cos(x)$', lw = 2.5, color = c2, linestyle = '-')
# plt.legend(loc = 'lower left')
# plt.title(r'$\sin(x)+\cos(x)$', fontdict=font1)
# plt.xlabel('time', fontdict=font2)
# plt.ylabel('Mag', fontdict=font2)
# plt.grid(axis='x')
# plt.tight_layout()
#
# plt.show()




# A = np.array([[3,2],[2,4]])
# a,b = LA.eig(A)
# b = pd.DataFrame(b, columns=['v1','v2'])
# print(tabulate(b))
# print(f'eigen values{a}')
#
# N = 10000
# np.random.seed(123)
# x = np.random.randn(2,N)
# y = np.matmul(A,np.tanh(np.matmul(A,x)))
#
# # Regression Analysis
# def objective(x,a,b):
#     return a * x + b
# x_f,y_f = y[0,:], y[1,:]
# popt, _ = curve_fit(objective, x_f, y_f)
# slope, intercept = popt
#
# x_p = np.linspace(-10,10,1000)
# y_p = slope * x_p + intercept
#
# print(f'The intercept of the fitted line is {intercept:.3f}')
# print(f'The slope of the fitted line is {slope:.3f}')
#
#
#
# plt.figure(figsize=(12,8))
# plt.subplot(1,2,1)
# plt.scatter(x[0,:],x[1,:], color = 'blue', label = 'raw-data')
# plt.title('raw data')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()
# plt.tight_layout()
# plt.axis('equal')
#
# b = b.values
# plt.subplot(1,2,2)
# plt.scatter(y[0,:],y[1,:], color = 'blue', label = 'transformed-data')
# origin = np.array([[0,0],[0,0]])
# plt.quiver(*origin, a[0]*b[:,0], a[1]*b[:,1], color = ['y','c'], scale = 12 )
# plt.plot(x_p,y_p, '--r', lw = 4)
# plt.title('transformed data')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()
# plt.tight_layout()
# plt.axis('equal')
#
# plt.show()

# %%
