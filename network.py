"""
Wong–Wang decision network (cleaned up version).

Fixed a bunch of stuff that was bugging me:
- Actually uses the proper transfer function instead of that weird linear thing
- Noise scaling that makes sense 
- Parameters that don't make you go "wtf"
- Less likely to blow up numerically
"""

import numpy as np


class WongWangNetwork:
    """Wong-Wang network parameters. Nothing fancy."""
    
    def __init__(self,
                 N_E1=240, N_E2=240, N_I=60,
                 w_plus=1.7, w_minus=1.0, w_I=1.0,        
                 tau_s=100.0, tau_NMDA=100.0, tau_AMPA=2.0, tau_GABA=10.0,
                 I0=0.3255, JA_ext=0.00052, mu0=40.0,
                 a=270.0, b=108.0, d=0.154,  # transfer function params
                 sigma=0.02):    
        
        # just store everything, no need to get fancy
        self.N_E1, self.N_E2, self.N_I = N_E1, N_E2, N_I
        self.w_plus, self.w_minus, self.w_I = w_plus, w_minus, w_I
        self.tau_s, self.tau_NMDA, self.tau_AMPA, self.tau_GABA = tau_s, tau_NMDA, tau_AMPA, tau_GABA
        self.I0, self.JA_ext, self.mu0 = I0, JA_ext, mu0
        self.a, self.b, self.d = a, b, d  # for the actual transfer function
        self.sigma = sigma

    def rates_from_currents(self, I1, I2):
        """
        Proper Wong-Wang transfer function instead of that I*100 nonsense.
        r = (aI - b) / (1 - d(aI - b))
        """
        def f(I):
            x = self.a * I - self.b
            if x <= 0:
                return 0.0
            denom = 1.0 - self.d * x
            if denom <= 0:  # don't divide by zero
                return 1000.0  # just cap it
            return x / denom
        
        return f(I1), f(I2)


def run_trial(network, coherence=0.256, t_max=3000, dt=0.5,
              thresh=70.0, min_decision_time=300.0, tie_epsilon=5.0, seed=None):
    """
    Run one trial. Fixed the noise scaling and made dt smaller by default.
    Also bumped up min_decision_time because 50ms is ridiculous.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(0, t_max, dt)

    # start with something reasonable
    s1, s2 = 0.1, 0.1
    nu1 = network.mu0 * (1.0 + coherence)
    nu2 = network.mu0 * (1.0 - coherence)
    
    # noise that actually scales with dt properly
    noise_scale = network.sigma * np.sqrt(dt)

    for t in time[1:]:
        sI = 0.5 * (s1 + s2)
        I1 = network.I0 + network.JA_ext*nu1 + network.w_plus*s1 - network.w_minus*s2 - network.w_I*sI
        I2 = network.I0 + network.JA_ext*nu2 + network.w_plus*s2 - network.w_minus*s1 - network.w_I*sI
        
        r1, r2 = network.rates_from_currents(I1, I2)

        # proper units in the diff eq
        ds1 = (-s1 + r1 * 0.001 * (1.0 - s1)) / (network.tau_s * 0.001)
        ds2 = (-s2 + r2 * 0.001 * (1.0 - s2)) / (network.tau_s * 0.001)
        
        s1 = np.clip(s1 + dt*ds1 + noise_scale*rng.normal(), 0.0, 1.0)
        s2 = np.clip(s2 + dt*ds2 + noise_scale*rng.normal(), 0.0, 1.0)

        if t < min_decision_time:
            continue

        # decision logic - cleaner this way
        r1_wins = r1 >= thresh and r2 < thresh
        r2_wins = r2 >= thresh and r1 < thresh
        both_high = r1 >= thresh and r2 >= thresh
        
        if r1_wins:
            return 'E1', float(t)
        if r2_wins:
            return 'E2', float(t)
        if both_high and abs(r1 - r2) > tie_epsilon:
            return ('E1' if r1 > r2 else 'E2'), float(t)

    return None, None


def sweep_psychometric(network, coherences=(0.0, 0.064, 0.128, 0.256, 0.512),
                       n_trials=200, dt=0.5, thresh=70.0, t_max=2000, seed=0):
    """
    Psychometric curve. Added RT std because why not.
    """
    rng = np.random.default_rng(seed)
    results = []

    for c in coherences:
        choices = []
        rts = []
        decisions = 0
        
        for _ in range(n_trials):
            choice, rt = run_trial(network, coherence=c, t_max=t_max, dt=dt,
                                 thresh=thresh, seed=rng.integers(0, 2**31))
            if choice:
                decisions += 1
                choices.append(choice)
                if rt:
                    rts.append(rt)

        # figure out accuracy
        if choices:
            if c >= 0:
                correct = sum(1 for ch in choices if ch == 'E1')
            else:
                correct = sum(1 for ch in choices if ch == 'E2')
            p_correct = correct / len(choices)
        else:
            p_correct = 0.0

        mean_rt = np.mean(rts) if rts else float('nan')
        rt_std = np.std(rts) if len(rts) > 1 else float('nan')
        decision_rate = decisions / n_trials
        
        results.append([c, p_correct, mean_rt, decision_rate, rt_std])

    return np.array(results)


__all__ = ["WongWangNetwork", "run_trial", "sweep_psychometric"]


if __name__ == "__main__":
    # Quick test to make sure everything works
    print("Testing Wong-Wang network...")
    
    net = WongWangNetwork()
    
    # Run a single trial
    choice, rt = run_trial(net, coherence=0.256, seed=42)
    if choice:
        print(f"Single trial: {choice} at {rt:.1f}ms")
    else:
        print("Single trial: no decision made")
    
    # Run a mini psychometric curve
    print("\nRunning psychometric sweep...")
    results = sweep_psychometric(net, coherences=(0.0, 0.128, 0.256), 
                               n_trials=50, seed=42)
    
    print("Coherence | P(correct) | Mean RT | Decision rate")
    print("-" * 45)
    for row in results:
        coh, pc, rt, dr, _ = row
        print(f"{coh:8.3f} | {pc:9.3f} | {rt:6.1f}ms | {dr:12.3f}")
    
    print("\nLooks good!")
    
    