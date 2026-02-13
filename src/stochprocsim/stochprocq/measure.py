"""
This module contains functions to compute various measures between probability distributions.
"""

import numpy as np
from scipy.stats import entropy

def kl_div(p:np.ndarray, q:np.ndarray) -> float:
    """
    Compute the Kullback-Leibler divergence between two probability distributions.

    Parameters:
    -------------------
    p: np.ndarray
        The first probability distribution.
    q: np.ndarray
        The second probability distribution.

    Return:
    ----------------
    kl: float
        The Kullback-Leibler divergence between the two distributions.
    """
    return np.sum(p * np.log(p / q), axis=-1)

def check_len(p:np.array, q:np.array):
    if len(p) != len(q):
        raise ValueError(
            f"""The shape of the two distributions are not consistent:
            {len(p)} and {len(q)}.""")
    return int(np.log2(len(p)))

def eval_diverge(
        p_dist:np.array,
        q_dist:np.array,
        his_steps:int = None
        ) -> float:
    """
    Evaluate the KL-divergnece function with respect to the **stationary distribution**

    Parameters
    ----------
    p_dist : np.array
        The true distribution.

    q_dist : np.array
        The predicted distribution.
    his_steps : int, optional
        Number of history steps for conditional divergence.
        If None, the full divergence is computed.
        The default is None.

    Returns
    -------
    divergence: float
        The KL-divergence value.
    """
    n = check_len(p_dist,q_dist)

    if his_steps is not None:
        p_dist_cond = p_dist.reshape((2**his_steps,-1)).copy()
        q_dist_cond = q_dist.reshape((2**his_steps,-1)).copy()

        diverge = 0
        for p,q in zip(p_dist_cond,q_dist_cond):
            if (p_his := np.sum(p)) > 1e-10:
                diverge += p_his * entropy(p/p_his, q/np.sum(q), base=2)
        return diverge/(n-his_steps)
    return entropy(p_dist, q_dist, base=2)

def eval_log_fid(
        p_dist:np.array,
        q_dist:np.array,
        his_steps:int = None
        ) -> float:
    """
    Evaluate the fidelity function with respect to the **stationary distribution**

    Parameters
    ----------
    p_dist : np.array
        The true distribution.

    q_dist : np.array
        The predicted distribution.
    his_steps : int, optional
        Number of history steps for conditional divergence.
        If None, the full divergence is computed.
        The default is None.

    Returns
    -------
    fidelity: float
        The fidelity value.
    """
    n = check_len(p_dist,q_dist)

    if his_steps is not None:
        p_dist_cond = p_dist.reshape((2**his_steps,-1)).copy()
        q_dist_cond = q_dist.reshape((2**his_steps,-1)).copy()

        fid = 0
        for p,q in zip(p_dist_cond,q_dist_cond):
            if p_his := np.sum(p) > 1e-10:
                a = np.sum((p/p_his)**2)
                b = np.sum((q/np.sum(q))**2)
                c = np.sum((p/p_his)*(q/np.sum(q)))
                fid += p_his * np.log2(c / np.sqrt(a*b))
        return fid/(n-his_steps)

    a = np.sum(p_dist**2)
    b = np.sum(q_dist**2)
    c = np.sum(p_dist*q_dist)
    return np.log2(c / np.sqrt(a*b))

def eval_nll(
        p_dist:np.array,
        q_dist:np.array,
        his_steps:int = None
        ) -> float:
    """
    Evaluate the negative log-likelihood function with respect to the **stationary distribution**

    Parameters
    ----------
    p_dist : np.array
        The true distribution.

    q_dist : np.array
        The predicted distribution.
    his_steps : int, optional
        Number of history steps for conditional divergence.
        If None, the full divergence is computed.
        The default is None.

    Returns
    -------
    nll: float
        The negative log-likelihood value.
    """
    n = check_len(p_dist,q_dist)

    if his_steps is not None:
        p_dist_cond = p_dist.reshape((2**his_steps,-1)).copy()
        q_dist_cond = q_dist.reshape((2**his_steps,-1)).copy()

        nll = 0
        for p,q in zip(p_dist_cond,q_dist_cond):
            if p_his := np.sum(p) > 1e-10:
                cond_p = p/p_his
                cond_q = q/np.sum(q)
                for pi,qi in zip(cond_p,cond_q):
                    nll += - p_his * pi * np.log2(qi)
        return nll/(n-his_steps)

    nll = 0
    for pi,qi in zip(p_dist,q_dist):
        nll += - pi * np.log2(qi)
    return nll

