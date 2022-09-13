import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from itertools import product
import numpy as np
from scipy.optimize import minimize
import xlwings as xw
#plt.rcParams['figure.figsize'] = [13, 10]

def combination_tree(n):
    '''generate a tree of all possible combination'''
    c = np.array([1,0])
    num_cols = 2
    for i in range(n-1):
        c1 = np.vstack((c,np.ones(num_cols)))
        cz = np.vstack((c,np.zeros(num_cols)))
        c = np.hstack((c1,cz))
        num_rows, num_cols = c.shape
    return np.transpose(c)

def pmf(c,p):
    '''generate the probabilities if each outcome defined in the probability tree'''
    try:
        num_rows, num_cols = c.shape
    except:
        num_rows = 2**len(p)
    P = np.ones(num_rows)
    for j in range(len(p)):
        if num_rows == 2: #single stock case
            P = np.multiply(P,(c*p[j]+(1-c)*(1-p[j])))
        else:
            P = np.multiply(P,(c[:,j])*p[j]+(1-c[:,j])*(1-p[j]))
    return P

def net_gain(b,a,c,f):
    '''calculate the array of net gains'''
    if len(b)==1:
        R=c*f*b-(1-c)*f*a
    else:
        R=np.matmul(c,np.multiply(f,b))-np.matmul(1-c,np.multiply(f,a))
    return R

def growth(R,P):
    g=1
    for i in range(len(P)):
        g=g*(1+R[i])**P[i]
    return g

def meanvar(R,P,t=0.20):
    m=np.sum(R*P)
    Rd=R[R<t]
    Pd=P[R<t]
    vd=np.sum(((Rd-t)**2)*Pd)/np.sum(Pd)
    return m,vd
    
def weighted_percentile(data, percents, weights=None):
    ''' percents in units of 1%
        weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind=np.argsort(data)
    d=data[ind]
    w=weights[ind]
    p=1.*w.cumsum()/w.sum()*100
    y=np.interp(percents, p, d)
    return y
    
def growth_wrapper(f,b,a,p):
    n=len(f)
    c=combination_tree(n)
    P=pmf(c,p)
    R=net_gain(b,a,c,f)
    g=growth(R,P)
    m,v=meanvar(R,P)
    return g,m,v

def gvis(n,fmax,b,a,p):
    '''histgram and growth curve for multiple, independant bets.
    histogram assuming weights are equal, fully invested'''

    f=np.ones(n)*1/n
    b=np.ones(n)*b
    a=np.ones(n)*a
    p=np.ones(n)*p
    
    c=combination_tree(n)
    P=pmf(c,p)
    R=net_gain(b,a,c,f)

    fig, ax = plt.subplots()
    N, bins, patches = plt.hist(R, 30, facecolor='b', alpha=0.75, weights=P)
    plt.xlabel('Returns')
    plt.ylabel('Probability')
    plt.title('Histogram Returns')
    plt.grid(True)
    plt.show()

    F = np.linspace(0.01, fmax, 100)
    g = np.zeros(len(F))

    for i in range(len(F)):
        f=np.ones(n)*F[i]
        g[i]=growth_wrapper(f,b,a,p)

    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()
    ax.plot(F, g, linewidth=2.0)
    ax.set(xlabel='stock weight, f', ylabel='portfolio growth (>1 good)')

    plt.show()
    
def weight_generator(n,Fmax=1,constraint='<'):
    if constraint=='<':
        n=n
    elif constraint=='=':
        n=n-1
        
    f=np.zeros(n)
    pt=np.zeros(n)
    for i in range(n):
        pt[i]=np.random.default_rng().uniform(low=0,high=Fmax)
    pts=np.insert(np.sort(pt),0,0)
    if constraint=='<':
        pass
    elif constraint=='=':
        pts=np.append(pts,Fmax)
    f=np.diff(pts)
    return f

def MCG(n,b,a,p,runs=2,Fmax=1,constraint='<'):
    '''Monte-Carlo with all the weights constrained by the maxium leverage'''
    f=np.zeros((runs,n))
    y=np.zeros(runs)
    m=np.zeros(runs)
    v=np.zeros(runs)
    for i in range(runs):
        f[i]=weight_generator(n,Fmax,constraint)
        y[i],m[i],v[i]=growth_wrapper(f[i],b,a,p)
    return y,m,v,f

def DoE(n,b,a,p,levels=4,fmax=0.2,Fmax=1):
    fmax=Fmax/round(Fmax/fmax)-0.00001 #rounding to make it a multiple of Fmax
    e = fmax/(levels-1)*np.array(list(product(range(levels), repeat=n)))
    f = e[np.sum(e,axis=1) <= Fmax]
    rows,cols = f.shape 
    y=np.zeros(rows)
    m=np.zeros(rows)
    v=np.zeros(rows)
    for i in range(rows):
        y[i],m[i],v[i]=growth_wrapper(f[i],b,a,p)
    return y,m,v,f

def objective(f,b,a,p,t=1):
    g,m,v = growth_wrapper(f,b,a,p)
    #objective = -1*((g-1)+0.2*(g-1)/v**0.5)
    objective = -1*((g-1)-t*v)
    return objective

def constraint(f,Fmax):
    return Fmax-np.sum(f)

def Opti(n,b,a,p,t,fmax=0.2,Fmax=0.7):
    fo=np.ones(n)*1/n*Fmax
    B=(0,fmax)
    bnds = (B,)*n
    c={'type':'eq','fun':constraint,'args':(Fmax,)}
    sol = minimize(objective,fo,method='SLSQP',jac='3-point',options={'ftol': 1e-10,'eps':1e-11},bounds=bnds,constraints=c, args=(b,a,p,t))
    #sol = minimize(objective,fo,method='trust-constr',bounds=bnds,constraints=c, args=(b,a,p,t))
    g,m,v = growth_wrapper(sol.x,b,a,p)
    return sol,g,m,v
    
def nbapf_from_xl():
    #xw..Range corresponds to the range on the active book and sheet
    sheet = xw.sheets.active
    n=int(sheet.range('B23').value)
    last = n+1
    b=sheet.range((2,7),(last,7)).value
    a=sheet.range((2,8),(last,8)).value
    p=sheet.range((2,5),(last,5)).value
    t=sheet.range('B27').value
    fmax=sheet.range('B29').value
    Fmax=sheet.range('B28').value
    return n,b,a,p,t,fmax,Fmax

def opti_to_xl():
    sheet = xw.sheets.active
    n,b,a,p,t,fmax,Fmax = nbapf_from_xl()
    sol,g,m,v = Opti(n=int(n),b=b,a=a,p=p,t=t,fmax=fmax,Fmax=Fmax)
    sheet.range('I2').options(transpose=True).value=sol.x
    sheet.range('B25').value=g-1
    sheet.range('B26').value=v**0.5
    return sol.x
    
def bub_to_xl(f):
    sheet = xw.sheets.active
    n,b,a,p,t,fmax,Fmax = nbapf_from_xl()
    fig, ax = plt.subplots()
    plt.scatter(np.array(a)*100,np.array(b)*100, s=f*10000, alpha=0.5)
    ax.set_xlabel(r'Downside [%]', fontsize=15)
    ax.set_ylabel(r'Upside [%]', fontsize=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    
    #labels
    last = n+1
    tickers=sheet.range((2,13),(last,13)).value
    for i in range(n):

        label = tickers[i]
        plt.annotate(label, # this is the text
                     (np.array(a)[i]*100,np.array(b)[i]*100), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,3), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center
                     

    sheet.pictures.add(fig, name='Bubble', update=True)

def fig_to_xl():
    sheet = xw.sheets.active
    n,b,a,p,t,fmax,Fmax = nbapf_from_xl()
    tvec = np.linspace(0, t, 10)
    fopt = np.zeros([len(tvec),n])
    gopt = np.zeros(len(tvec))
    vopt = np.zeros(len(tvec))
    
    for i in range(len(tvec)):
        sol,g,m,v = Opti(n,b,a,p,t=tvec[i],fmax=fmax,Fmax=Fmax)
        fopt[i]=sol.x
        gopt[i]=g
        vopt[i]=v
    
    #y,m,v,f=DoE(n=n,b=b,a=a,p=p,levels=3,fmax=fmax,Fmax=Fmax)
    fig, ax = plt.subplots()
    #ax.scatter(v**0.5*100, (y-1)*100,c=np.sum(f,axis=1),alpha=0.75,edgecolors='black')
    plt.plot(vopt**0.5*100, (gopt-1)*100,marker='o',markerfacecolor='white')
    ax.set_xlabel(r'Risk: Semi-Deviation from 20% Return [%]', fontsize=15)
    ax.set_ylabel(r'Return: 2-yr Growth [%]', fontsize=15)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    sheet.pictures.add(fig, name='Frontier', update=True)
    #plt.show()