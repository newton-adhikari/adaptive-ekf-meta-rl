#!/usr/bin/env python3
"""
CC-MetaEKF: One-command training pipeline.

runs all experiments, saves results + logs.

"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import time, argparse, json, os, sys, warnings, logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")


# ================================================================
# Setup
# ================================================================

def setup_device():
    if torch.cuda.is_available():
        dev = "cuda"
        name = torch.cuda.get_device_name()
        torch.backends.cudnn.benchmark = True
    else:
        dev = "cpu"
        name = f"{os.cpu_count()} CPU cores"
        torch.set_num_threads(os.cpu_count() or 1)
    return dev, name

DEVICE, DEVICE_NAME = setup_device()

def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"run_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger("ccmetaekf"), log_file

def log_and_save(results, name, output_dir):
    path = output_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return path


# ================================================================
# Extended Kalman Filter (EKF)
# ================================================================

class EKF6D:
    def __init__(self, dt=0.1):
        # Setting filter dimensions:
        # n = number of states
        # m = number of measurements
        self.n = 6
        self.m = 2
        self.dt = dt

        # Storing lower and upper Chi-square bounds
        # for checking filter consistency using NEES
        self.chi2_lb = 1.237
        self.chi2_ub = 14.449

        # Initializing performance metrics
        self.nees = 0.0
        self.nis = 0.0

        # Initializing innovation covariance matrix
        self.S = np.eye(2)

    def reset(self, x0, P0, Q, R):
        # Loading initial state estimate
        self.x = x0.copy().astype(np.float64)

        # Loading initial covariance matrix
        self.P = P0.copy().astype(np.float64)

        # Storing process noise covariance
        self.Q = Q.copy().astype(np.float64)

        # Storing measurement noise covariance
        self.R = R.copy().astype(np.float64)

        # Resetting consistency metrics
        self.nees = 0.0
        self.nis = 0.0

        # Resetting innovation covariance
        self.S = np.eye(2)

    def _F(self, x):
        # Computing Jacobian matrix of the motion model
        _, _, th, vx, vy, _ = x
        dt = self.dt

        F = np.eye(6)

        # Calculating position sensitivity to heading
        F[0,2] = (-vx*np.sin(th) - vy*np.cos(th)) * dt
        F[0,3] = np.cos(th) * dt
        F[0,4] = -np.sin(th) * dt

        F[1,2] = (vx*np.cos(th) - vy*np.sin(th)) * dt
        F[1,3] = np.sin(th) * dt
        F[1,4] = np.cos(th) * dt

        # Relating heading angle to angular velocity
        F[2,5] = dt

        return F

    def _f(self, x, u):
        # Predicting next state using the nonlinear motion model
        px, py, th, vx, vy, om = x
        dt = self.dt

        return np.array([
            px + vx*np.cos(th)*dt - vy*np.sin(th)*dt,  # Updating x position
            py + vx*np.sin(th)*dt + vy*np.cos(th)*dt,  # Updating y position
            th + om*dt,                                # Updating heading angle
            vx + u[0]*dt,                              # Updating x velocity
            vy + u[1]*dt,                              # Updating y velocity
            om + u[2]*dt                               # Updating angular velocity
        ])

    def predict(self, u):
        # Computing Jacobian at current state
        F = self._F(self.x)

        # Predicting next state estimate
        self.x = self._f(self.x, u)

        # Predicting covariance growth
        self.P = F @ self.P @ F.T + self.Q

        # Keeping covariance symmetric
        self.P = (self.P + self.P.T) / 2

    def update(self, z):
        # Creating measurement matrix
        # Measuring only x and y positions
        H = np.zeros((2,6))
        H[0,0] = 1
        H[1,1] = 1

        # Computing innovation (measurement error)
        nu = z - H @ self.x

        # Computing innovation covariance
        S = H @ self.P @ H.T + self.R

        # Computing Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Updating state estimate
        self.x = self.x + K @ nu

        # Updating covariance using Joseph form
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Keeping covariance symmetric
        self.P = (self.P + self.P.T) / 2

        # Storing innovation covariance
        self.S = S

        # Calculating NIS for measurement consistency checking
        self.nis = float(nu @ np.linalg.inv(S) @ nu)

        return nu

    def compute_nees(self, x_true):
        # Computing estimation error
        e = x_true - self.x

        try:
            # Calculating NEES for state consistency checking
            self.nees = float(e @ np.linalg.inv(self.P) @ e)
        except:
            # Using large penalty value if covariance inversion fails
            self.nees = 100.0

        return self.nees

    def is_consistent(self):
        # Checking whether NEES lies inside expected bounds
        return self.chi2_lb <= self.nees <= self.chi2_ub


# ================================================================
# 1D EKF Used During Phase 0
# Tracking position and velocity in one dimension
# ================================================================

class EKF1D:
    def __init__(self, dt=0.1):
        self.dt = dt

        # Defining constant velocity state transition model
        self.F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])

        # Defining measurement model
        # Measuring position only
        self.H = np.array([
            [1.0, 0.0]
        ])

    def reset(self, x0, P0, Q, R):
        # Loading initial state estimate
        self.x = x0.copy().astype(np.float64)

        # Loading initial covariance estimate
        self.P = P0.copy().astype(np.float64)

        # Storing process noise covariance
        self.Q = Q.copy().astype(np.float64)

        # Storing measurement noise covariance
        self.R = R.copy().astype(np.float64)

    def predict(self):
        # Predicting next state
        self.x = self.F @ self.x

        # Predicting covariance growth
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Keeping covariance symmetric
        self.P = (self.P + self.P.T) / 2

    def update(self, z):
        # Computing measurement residual
        nu = z - self.H @ self.x

        # Computing residual covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Computing Kalman gain
        K = (self.P @ self.H.T) / S[0,0]

        # Updating state estimate
        self.x = self.x + K[:,0] * nu[0]

        # Updating covariance using Joseph form
        I_KH = np.eye(2) - K @ self.H.reshape(1,2)
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Keeping covariance symmetric
        self.P = (self.P + self.P.T) / 2

        return nu[0], S[0,0]

    def nees(self, x_true):
        # Computing estimation error
        e = x_true - self.x

        try:
            # Calculating NEES for filter consistency evaluation
            return float(e @ np.linalg.inv(self.P) @ e)
        except:
            # Returning large value if inversion fails
            return 100.0
        

# ================================================================
# Tasks and Environments
# ================================================================

class Task:
    def __init__(self, Q_diag, R_diag, regime="stationary", ct=50):

        # Stored baseline process and measurement noise
        self.Q_base = np.array(Q_diag)
        self.R_base = np.array(R_diag)

        # Remembered what kind of chaos this task wanted
        self.regime = regime
        self.change_time = ct

        # Generated future noise values for abrupt changes
        # Basically preparing a surprise attack for the EKF
        self.Q_after = self.Q_base * np.random.choice([0.2, 5.0], size=len(Q_diag))
        self.R_after = self.R_base * np.random.choice([0.2, 5.0], size=len(R_diag))

    def get_noise(self, t):

        # Keeping life easy: noise never changed
        if self.regime == "stationary":
            return np.diag(self.Q_base), np.diag(self.R_base)

        # Noise suddenly woke up and chose violence
        elif self.regime == "abrupt" and t >= self.change_time:
            return np.diag(self.Q_after), np.diag(self.R_after)

        # Slowly increasing uncertainty over time
        elif self.regime == "drift":
            s = 1 + 0.005 * t
            return np.diag(self.Q_base * s), np.diag(self.R_base * s)

        # Falling back to default noise values
        return np.diag(self.Q_base), np.diag(self.R_base)


def sample_task(rng):

    # Sampled process noise from a log-uniform distribution
    q = np.exp(rng.uniform(np.log(0.005), np.log(0.2), 6))

    # Sampled measurement noise
    r = np.exp(rng.uniform(np.log(0.05), np.log(2.0), 2))

    # Randomly picked a world behavior
    regime = rng.choice(
        ["stationary", "abrupt", "drift"],
        p=[0.5, 0.3, 0.2]
    )

    # Created a fresh task for the agent to suffer through
    return Task(q, r, regime, rng.integers(30, 70))


class Env6D:

    def __init__(self, ep_len=100):

        # Stored episode length
        self.ep_len = ep_len

        # Created EKF instance
        self.ekf = EKF6D()

        # Set nominal covariance values
        self.Q_nom = np.eye(6) * 0.05
        self.R_nom = np.eye(2) * 0.5

        # Keeping recent innovation history
        self.ctx_len = 30

        # Number of innovations exposed to the policy
        self.n_innov = 5

        # Observation vector size
        self.obs_dim = self.n_innov*2 + 1 + 1 + 1 + 6 + 6 + 2

        # Action controls 6 Q values + 2 R values
        self.act_dim = 8

        # Created RNG for generating random nonsense responsibly
        self.rng = np.random.default_rng()

    def reset(self, task=None):

        # Generated a new task if none was supplied
        self.task = task or sample_task(self.rng)

        # Reset episode clock
        self.t = 0

        # Chose a trajectory type
        traj = self.rng.choice(["circle", "straight", "random"])
        self.traj_type = traj

        # Started with zero state
        self.x_true = np.zeros(6)

        # Spawned circular motion
        if traj == "circle":
            self.x_true = np.array([2, 0, np.pi/2, 0, 1.0, 0.5])

        # Spawned boring but predictable straight motion
        elif traj == "straight":
            self.x_true = np.array([0, 0, 0, 1, 0, 0])

        # Added initialization error because perfect sensors only exist in slides
        x0 = self.x_true + self.rng.normal(0, 0.2, 6)

        # Reset EKF state
        self.ekf.reset(
            x0,
            np.eye(6) * 0.5,
            self.Q_nom.copy(),
            self.R_nom.copy()
        )

        # Started innovation history
        self.innovs = [[0.0, 0.0]] * self.ctx_len

        # Started NEES history tracking
        self.nees_history = []

        return self._obs(), self._ctx()

    def step(self, action):

        # Converted actions into positive scaling factors
        alphas = np.clip(np.exp(action), 0.01, 100.0)

        # Updated EKF covariance guesses
        self.ekf.Q = np.diag(alphas[:6]) @ self.Q_nom
        self.ekf.R = np.diag(alphas[6:]) @ self.R_nom

        # Retrieved actual environment noise
        Q_true, R_true = self.task.get_noise(self.t)

        # Started with zero control input
        u = np.zeros(3)

        # Generated random controls for the random trajectory
        if self.traj_type == "random":
            u = self.rng.normal(0, 0.3, 3)

        # Propagated true state using real noise
        self.x_true = (
            self.ekf._f(self.x_true, u)
            + self.rng.multivariate_normal(np.zeros(6), Q_true)
        )

        # Generated noisy position measurement
        z = (
            self.x_true[:2]
            + self.rng.multivariate_normal(np.zeros(2), R_true)
        )

        # Running EKF predict-update cycle
        self.ekf.predict(u)
        nu = self.ekf.update(z)

        # Calculated consistency score
        nees = self.ekf.compute_nees(self.x_true)

        # Stored latest innovation
        self.innovs.append(nu.tolist())

        # Keeping only recent history
        self.innovs = self.innovs[-self.ctx_len:]

        self.t += 1

        # Checking whether episode finished
        done = self.t >= self.ep_len

        # Tracking NEES history for smoother rewards
        self.nees_history.append(nees)

        # Averaging recent consistency values
        avg_nees = np.mean(self.nees_history[-20:])

        # Measuring state estimation error
        rmse = float(
            np.sqrt(np.mean((self.x_true - self.ekf.x) ** 2))
        )

        # Rewarding NEES staying near target value
        # Less drama = more reward
        reward = -np.log1p(abs(avg_nees - 6))

        # Punishing large estimation errors
        reward -= 0.05 * min(rmse, 10.0)

        # EKF behaving itself earned bonus points
        if self.ekf.is_consistent():
            reward += 0.3

        # Running average staying in healthy range
        if 3.0 <= avg_nees <= 12.0:
            reward += 0.3

        return (
            self._obs(),
            self._ctx(),
            reward,
            done,
            {
                "nees": nees,
                "rmse": rmse,
                "consistent": self.ekf.is_consistent()
            }
        )

    def _obs(self):

        # Flattening recent innovations into one vector
        iv = np.array(
            self.innovs[-self.n_innov:]
        ).flatten()

        # Building observation features for the policy
        # Compressing huge values before they start causing drama
        return np.clip(
            np.concatenate([
                iv,

                # Normalized consistency metrics
                [self.ekf.nees / 6,
                 self.ekf.nis / 2],

                # Tracking covariance growth
                [np.log1p(max(np.trace(self.ekf.P), 0))],

                # State uncertainty
                np.log1p(np.maximum(np.diag(self.ekf.P), 0)),

                # Process noise estimate
                np.log1p(np.maximum(np.diag(self.ekf.Q), 0)),

                # Measurement noise estimate
                np.log1p(np.maximum(np.diag(self.ekf.R), 0))
            ]),
            -20,
            20
        ).astype(np.float32)

    def _ctx(self):

        # Returning full innovation history
        # Agent memory, because forgetting is bad
        return np.array(
            self.innovs,
            dtype=np.float32
        ).flatten()


# ================================================================
# Networks
# ================================================================

class STSIEEncoder(nn.Module):
    def __init__(self, innov_dim=2, filter_dim=4, latent=32, window=16, hop=4):
        super().__init__(); self.window=window; self.hop=hop; ch=16

        # making cnn layers for taking spectral features
        self.cnn=nn.Sequential(nn.Conv2d(innov_dim,ch,3,padding=1),nn.ReLU(),
            nn.Conv2d(ch,ch*2,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)))

        # making projection for spectral embedding
        self.embed=64; self.sp=nn.Linear(ch*2*16,self.embed)

        # making filter statistic embedding
        self.fe=nn.Sequential(nn.Linear(filter_dim,self.embed),nn.ReLU(),nn.Linear(self.embed,self.embed))

        # using attention for joining spectral and filter info
        self.ca=nn.MultiheadAttention(self.embed,4,batch_first=True)

        # making final latent feature output
        self.out=nn.Sequential(nn.Linear(self.embed+filter_dim,latent),nn.ReLU(),nn.Linear(latent,latent))

        # storing hann window for fft processing
        self.register_buffer("hann",torch.hann_window(window).float())

    def _spec(self, buf):
        B,L,D=buf.shape

        # checking if buffer is smaller than window
        if L < self.window:
            # adding zero padding for matching window size
            pad = torch.zeros(B, self.window - L, D, device=buf.device)
            buf = torch.cat([pad, buf], dim=1)
            L = self.window

        # calculating number of frames
        nf=max(1,(L-self.window)//self.hop+1)

        # making overlapping frames from buffer
        fr=torch.stack([buf[:,i*self.hop:i*self.hop+self.window] for i in range(nf)],1)

        # applying hann window on frames
        fr=fr*self.hann[None,None,:,None]

        # reshaping for fft calculation
        flat=fr.permute(0,1,3,2).reshape(-1,self.window)

        # calculating power spectrum
        fft=torch.fft.rfft(flat, dim=-1); pw=(fft.abs()**2)/self.window

        # returning log power spectrum
        return torch.log(pw.reshape(B,nf,D,self.window//2+1).permute(0,2,1,3)+1e-10)

    def forward(self, ib, fs):

        # extracting spectral features
        s=self._spec(ib); f=self.cnn(s).flatten(1)

        # making spectral token and filter token
        st=self.sp(f).unsqueeze(1); q=self.fe(fs).unsqueeze(1)

        # applying cross attention
        a,_=self.ca(q,st,st)

        # returning latent representation
        return self.out(torch.cat([a.squeeze(1),fs],-1))

class MLPEncoder(nn.Module):
    def __init__(self, ctx_dim, filter_dim=4, latent=32):
        super().__init__()

        # making simple mlp encoder
        self.net=nn.Sequential(nn.Linear(ctx_dim+filter_dim,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,latent))

    def forward(self, ib, fs):

        # joining inputs and making latent feature
        return self.net(torch.cat([ib.flatten(1),fs],-1))

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, encoder, hidden=256):
        super().__init__(); self.encoder=encoder; latent=32; inp=obs_dim+latent

        # making actor network for action generation
        self.actor=nn.Sequential(nn.Linear(inp,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())

        # making mean and std parameters
        self.mean=nn.Linear(hidden,act_dim); self.log_std=nn.Parameter(torch.zeros(act_dim)-0.5)

        # making reward critic network
        self.critic=nn.Sequential(nn.Linear(inp,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))

        # making cost critic network
        self.cost_critic=nn.Sequential(nn.Linear(inp,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))

    def forward(self, obs, ib, fs):

        # encoding innovation and filter stats
        z=self.encoder(ib,fs)

        # joining observation and latent feature
        x=torch.cat([obs,z],-1)

        # making actor hidden feature
        h=self.actor(x)

        # returning action distribution and critics
        return Normal(self.mean(h),self.log_std.exp().expand(obs.shape[0],-1)),self.critic(x),self.cost_critic(x)

    def act(self, obs, ib, fs):
        with torch.no_grad():

            # getting policy outputs
            d,v,cv=self.forward(obs,ib,fs)

            # sampling action from distribution
            a=d.sample()

            # calculating action log probability
            lp=d.log_prob(a).sum(-1)

        # returning numpy values
        return a.cpu().numpy(),lp.cpu().numpy(),v.cpu().numpy(),cv.cpu().numpy()

class PIDLag:
    def __init__(self,delta=0.15,kp=0.1,ki=0.008,kd=0.02,imax=5):

        # storing pid parameters
        self.delta=delta;self.kp=kp;self.ki=ki;self.kd=kd;self.imax=imax

        # initializing lagrangian values
        self.lam=0;self.integral=0;self.prev=0

    def update(self,vr):

        # calculating constraint error
        e=vr-self.delta

        # updating integral term
        self.integral=np.clip(self.integral+e,-self.imax,self.imax)

        # calculating derivative term
        d=e-self.prev; self.prev=e

        # updating lagrangian multiplier
        self.lam=max(0,self.kp*e+self.ki*self.integral+self.kd*d)

        return self.lam

    def reset(self):

        # resetting pid states
        self.lam=0;self.integral=0;self.prev=0


# ================================================================
# PPO Core
# ================================================================

def compute_gae(rew,val,cost,cval,done,gamma=0.99,lam=0.95):

    # creating advantage buffers
    T=len(rew); adv=np.zeros(T); cadv=np.zeros(T); lg=0; clg=0

    # going backward for gae calculation
    for t in reversed(range(T)):

        # getting next values
        nv=val[t+1] if t+1<len(val) else 0; ncv=cval[t+1] if t+1<len(cval) else 0

        # calculating reward advantage
        d=rew[t]+gamma*nv*(1-done[t])-val[t]; adv[t]=lg=d+gamma*lam*(1-done[t])*lg

        # calculating cost advantage
        cd=cost[t]+gamma*ncv*(1-done[t])-cval[t]; cadv[t]=clg=cd+gamma*lam*(1-done[t])*clg

    # returning returns and advantages
    return adv+np.array(val[:T]),adv,cadv+np.array(cval[:T]),cadv

def ppo_update(policy,opt,pid,ob,ib,fs,act,ret,adv,cret,cadv,logp,cost,
               use_constraint=True,clip=0.2,epochs=6,bs=512,max_kl=0.02):

    # getting batch size and device
    N=ob.shape[0]; dev=DEVICE

    # normalizing advantages
    adv=(adv-adv.mean())/(adv.std()+1e-8)

    # updating lagrangian multiplier
    lam=pid.update(cost.mean()) if use_constraint else 0.0

    # converting numpy arrays into tensors
    ot=torch.tensor(ob,dtype=torch.float32,device=dev)
    ibt=torch.tensor(ib,dtype=torch.float32,device=dev)
    fst=torch.tensor(fs,dtype=torch.float32,device=dev)
    at=torch.tensor(act,dtype=torch.float32,device=dev)
    rt=torch.tensor(ret,dtype=torch.float32,device=dev)
    adt=torch.tensor(adv,dtype=torch.float32,device=dev)
    crt=torch.tensor(cret,dtype=torch.float32,device=dev)
    cadt=torch.tensor(cadv,dtype=torch.float32,device=dev)
    lpt=torch.tensor(logp,dtype=torch.float32,device=dev)

    # running multiple ppo epochs
    for ep in range(epochs):

        # shuffling samples
        idx=torch.randperm(N,device=dev); kls=0; nb=0

        # making mini batches
        for s in range(0,N,bs):

            # selecting mini batch
            mb=idx[s:s+bs]; dist,val,cval=policy(ot[mb],ibt[mb],fst[mb])

            # calculating policy ratio
            lp=dist.log_prob(at[mb]).sum(-1); ratio=(lp-lpt[mb]).exp()

            # estimating kl divergence
            with torch.no_grad(): kls+=abs(((ratio-1)-(lp-lpt[mb])).mean().item()); nb+=1

            # calculating clipped surrogate objective
            s1=ratio*adt[mb]; s2=torch.clamp(ratio,1-clip,1+clip)*adt[mb]

            # calculating total ppo loss
            loss=(-torch.min(s1,s2).mean()+0.5*(rt[mb]-val.squeeze()).pow(2).mean()
                  +0.5*(crt[mb]-cval.squeeze()).pow(2).mean()-0.01*dist.entropy().sum(-1).mean())

            # adding constraint penalty
            if use_constraint: loss+=lam*(ratio*cadt[mb]).mean()

            # doing gradient updating
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(),0.5); opt.step()

        # stopping early if kl is large
        if nb>0 and kls/nb>max_kl: break

    # returning training statistics
    return {"pi":loss.item(),"lam":lam,"viol":float(cost.mean())}

def collect(env,policy,n_steps,action_clip=5.0):

    # making rollout storage lists
    ol,il,fl,al,rl,cl,dl,ll,vl,cvl=[],[],[],[],[],[],[],[],[],[]

    # resetting environment
    obs,ctx=env.reset()

    # collecting rollout samples
    for _ in range(n_steps):

        # getting innovation buffer
        ib=np.array(env.innovs,dtype=np.float32)

        # getting filter statistics
        fs=np.array([env.ekf.nees/6,env.ekf.nis/2,np.log1p(np.trace(env.ekf.P)),np.log1p(env.ekf.S[0,0])],dtype=np.float32)

        # converting inputs into tensors
        ot=torch.tensor(obs,dtype=torch.float32,device=DEVICE).unsqueeze(0)
        ibt=torch.tensor(ib,dtype=torch.float32,device=DEVICE).unsqueeze(0)
        fst=torch.tensor(fs,dtype=torch.float32,device=DEVICE).unsqueeze(0)

        # sampling action from policy
        a,lp,v,cv=policy.act(ot,ibt,fst); a=a.squeeze();lp=lp.squeeze();v=v.squeeze();cv=cv.squeeze()

        # stepping environment with clipped action
        nobs,nctx,rew,done,info=env.step(np.clip(a,-action_clip,action_clip))

        # storing rollout data
        ol.append(obs);il.append(ib);fl.append(fs);al.append(a);rl.append(rew)

        # storing cost and value info
        cl.append(0.0 if info["consistent"] else 1.0);dl.append(float(done));ll.append(lp);vl.append(v);cvl.append(cv)

        # resetting or moving next state
        obs,ctx=(env.reset() if done else (nobs,nctx))

    # returning collected rollout arrays
    return tuple(np.array(x) for x in [ol,il,fl,al,rl,cl,dl,ll,vl,cvl])

def evaluate(env,policy,tasks,n_ep=5):

    # storing evaluation results
    results=[]

    # running all evaluation tasks
    for task in tasks:
        for _ in range(n_ep):

            # resetting environment with task
            obs,ctx=env.reset(task=task); nl=[]; done=False

            # running episode
            while not done:

                # getting innovation buffer
                ib=np.array(env.innovs,dtype=np.float32)

                # getting filter statistics
                fs=np.array([env.ekf.nees/6,env.ekf.nis/2,np.log1p(np.trace(env.ekf.P)),np.log1p(env.ekf.S[0,0])],dtype=np.float32)

                # converting inputs into tensors
                ot=torch.tensor(obs,dtype=torch.float32,device=DEVICE).unsqueeze(0)
                ibt=torch.tensor(ib,dtype=torch.float32,device=DEVICE).unsqueeze(0)
                fst=torch.tensor(fs,dtype=torch.float32,device=DEVICE).unsqueeze(0)

                # taking mean action for evaluation
                with torch.no_grad(): d,_,_=policy(ot,ibt,fst); a=d.mean.squeeze().cpu().numpy()

                # stepping environment
                obs,ctx,_,done,info=env.step(np.clip(a,-5,5))

                # storing nees values
                nl.append(info["nees"])

            # calculating episode metrics
            na=np.array(nl); results.append({"nees":np.mean(na),"cons":float(np.mean((na>=1.237)&(na<=14.449))),"rmse":np.mean([info["rmse"]])})

    # returning averaged evaluation metrics
    return (np.mean([r["nees"] for r in results]),np.mean([r["cons"] for r in results]),
            np.mean([r["rmse"] for r in results]))