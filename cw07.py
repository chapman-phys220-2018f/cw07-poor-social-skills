#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
# Name: Trevor Kling
# Student ID: 002270716
# Email: kling109@mail.chapman.edu
# Course: PHYS220/MATH220/CPSC220 Fall 2018
# Assignment: CW07
###

"""Classwork 07
This classwork introduces numpy arrays and compares their performance to
python lists.
"""

import math
import numpy as np

def gen_gaussian_list(a, b, n=1000):
    """gen_gaussian_list(a, b, n=1000)
    Generate a discrete approximation of a Gaussian function, including its
    domain and range, stored as a pair of vanilla python lists.
    
    Args:
        a (float) : Lower bound of domain
        b (float) : Upper bound of domain
        n (int, optional) : Number of points in domain, defaults to 1000.
    
    Returns:
        (x, g) : Pair of lists of floats
            x  : [a, ..., b] List of n equally spaced floats between a and b
            g  : [g(a), ..., g(b)] List of Gaussian values matched to x
    """
    dx = (b-a)/(n-1)                         # spacing between points
    x = [a + k*dx for k in range(n)]         # domain list
    
    # Local implementation of a Gaussian function
    def gauss(x):
        return (1/math.sqrt(2*math.pi))*math.exp(-x**2/2)
    
    g = [gauss(xk) for xk in x]                  # range list
    return (x, g)


def gen_gaussian_array(a, b, n=1000):
    """gen_gaussian_array(a, b, n=1000)
    Generate a discrete approximation of a Gaussian function, including its
    domain and range, stored as a pair of numpy arrays.
    
    Args:
        a (float) : Lower bound of domain
        b (float) : Upper bound of domain
        n (int, optional) : Number of points in domain, defaults to 1000.
    
    Returns:
        (x, g) : Pair of numpy arrays of float64
            x  : [a, ..., b] Array of n equally spaced float64 between a and b
            g  : [g(a), ..., g(b)] Array of Gaussian values matched to x
    """
    dx = (b-a)/(n-1)
    x = np.arange(0, n)
    def arr_gen(i):
        return a + i*dx

    xarr = np.vectorize(arr_gen)
    generatedArray = xarr(x)

    def gauss(v):
        return (1/math.sqrt(2*math.pi))*math.exp(-v**2/2)

    garr = np.vectorize(gauss)
    g = garr(generatedArray)
    return((generatedArray, g))

def gen_sinc_list(a, b, n=1000):
    """
    Generates a list based on the "sinc" function.  Based on the template for the Gaussian function above.

    Args:
      a (float): Lower bound of domain
      b (float): Uper bound of domain
      n (int): number of points in the domain, with a default value of 1000.

    Returns:
     (generatedList, sincList):
       generatedList [float] : [a,...,b] A list of floats used as the inputs to the function.
       sincList [float] : [sinc(a),...,sinc(b)] A list of floats produced by running the inputs through the sinc program.
    """
    dx = (b-a)/(n-1)
    x = np.arange(0, n)

    generatedList = [a + kx * dx for kx in x]

    sincList = [np.divide(np.sin(t), t, out=None, where=t!=0) for t in generatedList]
    return (generatedList, sincList)

def gen_sinc_array(a, b, n=1000):
    """
    Generates an array based on the "sinc" function.  Based on the template for the Gaussian function above.

    Args:
      a (float): Lower bound of domain
      b (float): Uper bound of domain
      n (int): number of points in the domain, with a default value of 1000.

    Returns:
     (generatedArray, sincArr):
       generatedArray [float] : [a,...,b] An array of floats used as the inputs to the function.
       sincArr [float] : [sinc(a),...,sinc(b)] An array of floats produced by running the inputs through the sinc program.
    """
    dx = (b-a)/(n-1)
    x = np.arange(0, n)
    def arr_gen(i):
        return a + i*dx

    xarr = np.vectorize(arr_gen)
    generatedArray = xarr(x)

    sinGen = np.vectorize(np.sin)
    sinArray = sinGen(generatedArray)
    sincArr = np.arange(0, n)

    sincArr = np.divide(sinArray, generatedArray, out=None, where=generatedArray!=0)

    return (generatedArray, sincArr)

def gen_sinf_list(a, b, n=1000):
    """
    Generates a list based on the "sinf" function.  Based on the template for the Gaussian function above.

    Args:
      a (float): Lower bound of domain
      b (float): Uper bound of domain
      n (int): number of points in the domain, with a default value of 1000.

    Returns:
     (generatedList, sinfList):
       generatedList [float] : [a,...,b] A list of floats used as the inputs to the function.
       sinfList [float] : [sinc(a),...,sinc(b)] A list of floats produced by running the inputs through the sinf program.
    """
    dx = (b-a)/(n-1)
    x = np.arange(0, n)

    generatedList = [a + kx * dx for kx in x]

    def sinf(t):
        return np.sin(np.divide(1, t, out=None, where=t!=0))

    sinfList = [sinf(kx) for kx in generatedList]
    return (generatedList, sinfList)

def gen_sinf_array(a, b, n=1000):
    """
    Generates an array based on the "sinf" function.  Based on the template for the Gaussian function above.

    Args:
      a (float): Lower bound of domain
      b (float): Uper bound of domain
      n (int): number of points in the domain, with a default value of 1000.

    Returns:
     (generatedArray, sinfArray):
       generatedArray [float] : [a,...,b] An array of floats used as the inputs to the function.
       sinfArray [float] : [sinc(a),...,sinc(b)] An array of floats produced by running the inputs through the sinf program.
    """
    dx = (b-a)/(n-1)
    x = np.arange(0, n)

    def arr_gen(i):
        return a + i*dx

    xarr = np.vectorize(arr_gen)
    generatedArray = xarr(x)

    def sinf(t):
        return np.sin(np.divide(1, t, out=None, where=t!=0))

    sinfGen = np.vectorize(sinf)
    sinfArray = sinfGen(generatedArray)
    return (generatedArray, sinfArray)


def main(a,b,n=1000):
    """main(a, b, n=1000)
    Main function for command line operation. Prints result of Gaussian to screen.
    
    Args:
        a (float) : Lower bound of domain
        b (float) : Upper bound of domain
        n (int, optional) : Number of points in domain, defaults to 1000.
    
    Returns:
        None
    
    Effects:
        Prints Gaussian to screen.
    """
    # You can unpack tuples as return values easily
    x, g = gen_gaussian_list(a,b,n)
    # The zip function takes two lists and generates a list of matched pairs
    for (xk, gk) in zip(x, g):
        # The format command replaces each {} with the value of a variable
        print("({}, {})".format(xk, gk))


# Protected main block for command line operation
if __name__ == "__main__":
    # The sys module contains features for running programs
    import sys
    # The sys.argv list variable contains all command line arguments
    #    sys.argv[0] is the program name always
    #    sys.argv[1] is the first command line argument, etc
    if len(sys.argv) == 4:
        a = float(sys.argv[1])
        b = float(sys.argv[2])
        n = int(sys.argv[3])
        main(a,b,n)
    elif len(sys.argv) == 3:
        a = float(sys.argv[1])
        b = float(sys.argv[2])
        main(a,b)
    else:
        print("Usage: cw07.py a b [n]")
        print("  a : float, lower bound of domain")
        print("  b : float, upper bound of domain")
        print("  n : integer, number of points in domain")
        exit(1)

