"""
StochProcQ: A Python package for stochastic processes and queueing theory.
"""

from .Models.renewal import get_uniform_renewal, get_bio_renewal
from .hmm import HiddenMarkovModel, MPS

def main():
    """
    Main function to run the package.
    """
    print("StochProcQ: A Python package for (quantum) stochastic processes.")
    print("Available models:")
    print("1. Uniform Renewal Process")
    print("2. Bio-Renewal Process")
