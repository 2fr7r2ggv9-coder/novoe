import numpy as np, wave, sys
SR=44100; DUR=15.0; N=int(SR*DUR); r=np.random.default_rng(3)
out=np.zeros((N,2),np.float32)
T0=105.45; L=lambda x:x-T0
def add(sig,t0,pan=0.0,g=1.0):
    i=int(t0*SR); j=min(N,i+len(sig))
    if j<=i: return
    s=sig[:j-i]*g
    out[i:j,0]+=s*(1-max(0,pan)); out[i:j,1]+=s*(1+min(0,pan))
def onepole(x,a):
    y=np.zeros_like(x); acc=0.0
    for k in range(len(x)): acc+=a*(x[k]-acc); y[k]=acc
    return y
def sizzle(d=0.9):
    n=int(d*SR); x=r.normal(0,1,n); x=np.diff(x,prepend=0); env=np.exp(-np.arange(n)/SR*4.5)*(1-np.exp(-np.arange(n)/SR*200))
    return (x*env*0.22).astype(np.float32)
def thump(f=70,d=0.35):
    n=int(d*SR); t=np.arange(n)/SR; f_=f*(1+1.5*np.exp(-t*30))
    ph=2*np.pi*np.cumsum(f_)/SR; return (np.sin(ph)*np.exp(-t*12)*0.6+r.normal(0,1,n)*np.exp(-t*80)*0.15).astype(np.float32)
def whoosh(d=0.8,up=True):
    n=int(d*SR); x=r.normal(0,1,n).astype(np.float32); t=np.arange(n)/n
    a=(0.02+0.25*(t if up else 1-t)); y=np.zeros(n,np.float32); acc=0.0
    for k in range(n): acc+=a[k]*(x[k]-acc); y[k]=acc
    env=np.sin(np.pi*t)**2
    return (y*env*0.9).astype(np.float32)
def crackle(d):
    n=int(d*SR); y=np.zeros(n,np.float32); t=np.arange(n)/SR
    dens=60+250*(t/d)
    k=0.0
    for i in range(int(d*300)):
        p=r.uniform(0,d)
        if r.random()<dens[int(p*SR)-1]/310:
            m=int(r.uniform(0.001,0.006)*SR); s=int(p*SR)
            if s+m<n: y[s:s+m]+=r.normal(0,1,m)*np.exp(-np.arange(m)/(m/4))*r.uniform(0.2,0.8)
    rumble=onepole(r.normal(0,1,n).astype(np.float32),0.01)*3
    env=np.minimum(1,t/0.3)*np.minimum(1,(d-t)/0.3)
    return ((y*0.5+rumble*0.4)*env).astype(np.float32)
def boom(d=1.4):
    n=int(d*SR); t=np.arange(n)/SR; ph=2*np.pi*np.cumsum(48*(1+0.8*np.exp(-t*8)))/SR
    return (np.sin(ph)*np.exp(-t*3)*0.75+onepole(r.normal(0,1,n).astype(np.float32),0.05)*np.exp(-t*6)*1.2).astype(np.float32)
# events
add(sizzle(),L(105.68),-0.3,0.9); add(sizzle(),L(106.24),0.2,0.9)
add(whoosh(1.1,True),L(106.55),0,0.5)               # crane up
for h in [L(110.44),L(111.08)]: add(thump(60,0.5),h,0,0.6); add(whoosh(0.6,False),h,0,0.35)
add(sizzle(0.7)*0.6,L(112.50),-0.4,0.8)              # coal dies
add(crackle(1.6),L(113.40),-0.3,1.1)               # leaf burns
add(whoosh(0.5,True),L(115.02),0,0.6)               # embers converge
add(boom(),L(115.47),0,0.7)
add(thump(85,0.3),L(116.68),0,0.7)
out=np.clip(out,-1,1)
with wave.open(sys.argv[1],'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((out*32767).astype(np.int16).tobytes())
print("sfx ok", np.abs(out).max())
