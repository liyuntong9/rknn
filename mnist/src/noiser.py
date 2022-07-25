import numpy as np


def rr_simulator(ep, delta, n, v):
    # ep: privacy budget
    # delta: sensitivity
    # n: number of clients
    # v: a true count
    # return unbiased estimation of counts after RR
    unitep = ep/(2*delta)
    tpr = np.exp(unitep)/(np.exp(unitep)+1)
    fpr = 1.0 - tpr
    observes = np.random.binomial(v, tpr)+np.random.binomial(n-v, fpr)
    estimator = (observes-n*fpr)/(tpr-fpr)

    return estimator


def collision_simulator(ep, delta, n, v):
    # ep: privacy budget
    # delta: sensitivity
    # n: number of clients
    # v: a true count
    # return unbiased estimation of counts after Collision mechanism
    l = 2*delta-1+delta*np.exp(ep)
    tpr = np.exp(ep)/(delta*np.exp(ep)+l-delta)
    fpr = 1/l
    observes = np.random.binomial(v, tpr)+np.random.binomial(n-v, fpr)
    estimator = (observes-n*fpr)/(tpr-fpr)

    return estimator

def rr_mechanism(ep, delta, A):
    # ep: privacy budget
    # delta: sensitivity
    # A: a delta-hot binary vector
    unitep = ep/(2*delta)
    tpr = np.exp(unitep)/(np.exp(unitep)+1)
    fpr = 1.0 - tpr
    observes = []
    for v in A:
        if np.random.random() > tpr:
            observes.append(1-v)
        else:
            observes.append(v)
    observes = np.array(observes)
    estimator = (observes-fpr)/(tpr-fpr)

    return estimator

def collision_mechanism(ep, delta, A):
    # ep: privacy budget
    # delta: sensitivity
    # A: a delta-hot binary vector
    d = len(A)
    l = int(2*delta-1+delta*np.exp(ep))
    omega = delta*np.exp(ep)+l-delta
    tpr = np.exp(ep)/(delta*np.exp(ep)+l-delta)
    fpr = 1/l

    z = 0

    seed = np.random.randint(0, 100000)

    # vector to set
    As = []
    for i,v in enumerate(A):
        if v != 0:
           As.append(i)

    # get collisiion set
    collisions = []
    for i in As:
        np.random.seed(seed+i)
        hv = np.random.randint(0, l)
        collisions.append(hv)
    collisions = list(set(collisions))

    #sample
    np.random.seed(None)
    ur = np.random.random()
    a = 0
    for i in range(l):
        if i in collisions:
            a += np.exp(ep)/omega
            if a > ur:
                z = i
                break
        else:
            a += (omega-len(collisions)*np.exp(ep))/((l-len(collisions))*omega)
            if a > ur:
                z = i
                break

    (z, seed) # the public message
    # decode
    pub = np.zeros(d, dtype=int)
    for i in range(0, d):
        np.random.seed(seed+i)
        v = np.random.randint(0, l)
        if v == z:
            pub[i] = 1

    estimator = (pub-fpr)/(tpr-fpr)

    return estimator



def multiplicative_mechanism(ep, T, Y, k, r=1, split=False):
    # ep: privacy budget
    # T: k-hot binary vector
    # Y: r-hot binary vector
    eTY = None
    if not split:
        # concatenative
        TY = T+Y
        eTY = collision_mechanism(ep, k+r, TY)
    else:
        # splitting
        eT = collision_mechanism(ep/2, k, T)
        eY = collision_mechanism(ep/2, r, Y)
        eTY = np.concatenate((eT, eY), axis=0)

    eT = np.array(eTY[0:len(T)]).reshape((len(T), 1))
    eY = np.array(eTY[len(T):len(T)+len(Y)]).reshape((1, len(Y)))
    eA = np.matmul(eT, eY)
    estimator = eA.flatten()

    return estimator





