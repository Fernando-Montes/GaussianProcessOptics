#!/usr/bin/env python

import sys, math
import os, shutil, signal
#import commands
import subprocess as commands
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import multiprocessing
import itertools
import timeit

# Function that runs cosy given field gradients and outputs resolution at FP3. 
# Output is written in file temp-results
from cosy import cosyrun 

# Functions to display and save results 
# from Figures import *

# Hyper-parameter
theta_nom = 0.004 # Kernel parameter
eps = 0.02 # Acquisition function (probability/expectation of improvement) parameter
num_points = 100000 # Number of points to sample

# Nominal magnetic field values
q1s_nom = -0.39773
q2s_nom = 0.217880+0.001472    
q3s_nom = 0.242643-0.0005+0.000729 
q4s_nom = -0.24501-0.002549 
q5s_nom = 0.1112810+0.00111 
q6s_nom = 0.181721-0.000093+0.00010-0.000096 
q7s_nom = -0.0301435+0.0001215 

# Returns kernel
def kernel( x1, x2, theta1, theta2) :
    x1 = np.asarray(x1).reshape(-1)
    x2 = np.asarray(x2).reshape(-1)
    return np.exp( -1/(np.power(theta1,2)+np.power(theta2,2))*np.dot(x1-x2,x1-x2) ) 

# Returns vector (column) k at position x (vector has information at position x due to observations) 
def k(x, x_observed, theta) :
    num_observations = np.shape(x_observed)[1]
    k = np.zeros(shape=(num_observations,1)) #column
    for i in range(0, num_observations):
        k[i][0] = kernel(x, x_observed[:,i], theta[i], theta_nom)
    return np.asmatrix(k)

# Returns covariance matrix with information from observations
def K(x_observed, theta):
    num_observations = np.shape(x_observed)[1]
    # Constructing Kernel observations
    K = np.zeros(shape=(num_observations,num_observations))
    for i in range(0, num_observations):
        for j in range(0, num_observations):
            K[i][j] = kernel( x_observed[:,i], x_observed[:,j], theta[i], theta[j] )
    return np.asmatrix(K)

# Returns reduced observations list and new covariance matrix
def reduce(x_observed, f_observed, theta, imaxf):
    eliminate = 0 # flag to know if observations have been reduced
#    ratiomin = 10 # something very large. Ratio of distance/theta_distance
#    dmin = 0
#    imin = 0
#    jmin = 0
#    num_observations = np.shape(x_observed)[1]
#    if num_observations > 30:
#        for i in range(0, num_observations-10):    # leave last 10 observations untouched
#            for j in range(i, num_observations-10):
#                dd = distPoints(x_observed, i, j)
#                td = theta[i]+theta[j]  # theta_distance
#                if dd/td < ratiomin and i!=j:
#                    ratiomin = dd/td
#                    dmin = dd
#                    imin = i
#                    jmin = j
#        print('i=%d, j=%d, dmin=%f, ratiomin=%f' %(imin,jmin,dmin,ratiomin))
#        if ratiomin < 1.0 and imin!=imaxf and jmin!=imaxf:  # Only eliminating observation if they are closer than theta_distance and not max observation
#            eliminate = 1
#            x1 = np.asarray( x_observed[:,imin] ).reshape(-1)
#            x2 = np.asarray( x_observed[:,jmin] ).reshape(-1)
#            # Calculating new point information
#            xnew = (x1+x2)/2
#            fnew = (f_observed[imin] + f_observed[jmin])/2
#            thetanew = dmin + theta[imin]/2 + theta[jmin]/2
#            print('Eliminating data point with i=%d, j=%d, dmin=%f, ratiomin=%f, new number of observations = %d' %(imin,jmin,dmin,ratiomin,num_observations-1))
#            x_observed = np.delete(x_observed, (imin), axis=1) # Deleting observation point
#            x_observed = np.delete(x_observed, (jmin-1), axis=1) # Deleting observation point
#            x_observed = np.hstack((x_observed,np.transpose(np.asmatrix(xnew))))  # Adding replacement point
#            f_observed = np.delete(f_observed, (imin), axis=0) # Deleting observation point
#            f_observed = np.delete(f_observed, (jmin-1), axis=0) # Deleting observation point
#            f_observed = np.concatenate((f_observed, fnew), axis=0) # Adding replacement point
#            theta =      np.delete(theta, (imin), axis=0) # Deleting observation point
#            theta =      np.delete(theta, (jmin-1), axis=0) # Deleting observation point
#            theta =      np.concatenate((theta, thetanew), axis=0) # Adding replacement point
    KK = K(x_observed, theta)  # Calculate new covariance matrix
    return x_observed, f_observed, theta, KK, eliminate

# returns distance between the latest observations
def distPoints(x_observed, i, j):
    x1 = np.asarray( x_observed[:,i] ).reshape(-1)
    x2 = np.asarray( x_observed[:,j] ).reshape(-1)
    dist = np.sqrt(np.dot(x1-x2,x1-x2))     
    return dist 

# Returns mean mu at position x
def mu( k, KInv, f_observed ):
    return np.transpose(k)*KInv*f_observed

# Returns sigma at position x
def sig( k, KInv ):
    tmp = 1- np.transpose(k)*KInv*k 
    if tmp[0,0] <= 0:
        return np.asmatrix([[0.0001]])
    else:
        return np.sqrt(tmp)

# Returns probability of improvement
def PI(mean, fxmax, eps, sigma):
    return norm.cdf((mean-fxmax-eps)/sigma)

# Return expectation of improvement
def EI(mean, fxmax, eps, sigma):
    zz = (mean-fxmax-eps)/sigma
    return (mean-fxmax-eps)*norm.cdf(zz)+sigma*norm.pdf(zz)
	
# Sample the PI/EI over phase space
def samplePS(num_points, fxmax) :
    PImax = 0
    for j in range(0, num_points):
        x = np.asmatrix( [ [random.uniform(q1s_nom*0.5, q1s_nom*1.5)],[random.uniform(q2s_nom*0.5, q2s_nom*1.5)], [random.uniform(q3s_nom*0.5, q3s_nom*1.5)], [random.uniform(q4s_nom*0.5, q4s_nom*1.5)], [random.uniform(q5s_nom*0.5, q5s_nom*1.5)], [random.uniform(q6s_nom*0.5, q6s_nom*1.5)], [random.uniform(q7s_nom*0.5, q7s_nom*1.5)]  ] )  # column
        kk = k(x[:,0], x_observed, theta) 
        mean = mu( kk, KInv, f_observed )[0,0]
        sigma = sig( kk, KInv )[0,0]
        #PIx = PI(mean, fxmax, eps, sigma)
        PIx = EI(mean, fxmax, eps, sigma)
        #if j%50 == 0:
        #	print j
        if PIx > PImax:
            PImax = PIx
            xPImax = x
    f = open('temp-sampling.txt','a') 	# Writing to sampling file best case
    f.write( '{0: .6f} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f} {6:.6f} {7:.6f}\n' .format(xPImax[0,0], xPImax[1,0], xPImax[2,0], xPImax[3,0], xPImax[4,0], xPImax[5,0], xPImax[6,0], PImax) )
    return 0


# Start of main ------------------------------------------------------------------------------------------------------------

start = int(sys.argv[1]) # flag to start from scratch (0) or to use previous data (1)
num_steps = int(sys.argv[2]) # Number of steps to do calculation
startTime = timeit.default_timer()

if start==0:
    # Removing files from older runs
    cmd = 'rm -f results_resolution.txt'
    failure, output = commands.getstatusoutput(cmd)
    # Removing files from older runs
    cmd = 'rm -f observations_*.txt'
    failure, output = commands.getstatusoutput(cmd)
    # Starting point 
    #x_observed = np.asmatrix( [[q2s_nom*0.9], [q3s_nom*0.9], [q4s_nom*0.9], [q5s_nom*0.9], [q6s_nom*0.9], [q7s_nom*0.9]] )  #column
    x_observed = np.asmatrix( [ [random.uniform(q1s_nom*0.5, q1s_nom*1.5)], [random.uniform(q2s_nom*0.5, q2s_nom*1.5)], [random.uniform(q3s_nom*0.5, q3s_nom*1.5)], [random.uniform(q4s_nom*0.5, q4s_nom*1.5)], [random.uniform(q5s_nom*0.5, q5s_nom*1.5)], [random.uniform(q6s_nom*0.5, q6s_nom*1.5)], [random.uniform(q7s_nom*0.5, q7s_nom*1.5)]  ] ) 
    f_observed = np.asmatrix( [[]] )  #column
    theta      = np.asmatrix( [[]] )  #column
    startN = 0
    newdistPoints = 0
    num_observations = 1
    fxmax = 0
    eliminate = 0
else:
    x_observed = np.asmatrix(np.loadtxt('observations_x_observed.txt'))
    f_observed = np.asmatrix(np.loadtxt('observations_f_observed.txt'))
    theta =      np.asmatrix(np.loadtxt('observations_theta.txt'))
    N =np.shape(x_observed)[1]
    startN = N-1
    fxmax = np.max(f_observed)
    imaxf = np.argmax(np.transpose(f_observed))

if __name__ == "__main__":
    for i in range(startN,startN+num_steps):
	
        print("step %d" % i)
        if not(startN!=0 and i==startN):   # if starting from scratch
            print( 'Distance between two last points [theta_units] = %f' %newdistPoints)        
            q1s = x_observed[0,num_observations-1]
            q2s = x_observed[1,num_observations-1]
            q3s = x_observed[2,num_observations-1]
            q4s = x_observed[3,num_observations-1]
            q5s = x_observed[4,num_observations-1]
            q6s = x_observed[5,num_observations-1]
            q7s = x_observed[6,num_observations-1]

            cosyrun( q1s, q2s, q3s, q4s, q5s, q6s, q7s )  # Running cosy calculation
            print('time (sec) after cosy: %f' % (timeit.default_timer() - startTime))  
            f = open('temp-results','r')  # Opening temp file with results
            resol = f.readline()
            f.close()
            f = open('results_resolution.txt','a') 	# Writing results file: q2s, resolution
            f_observed = np.concatenate((f_observed, [[float(resol)/1000]]), axis=1) 
            theta      = np.concatenate((theta,      [[theta_nom]]),         axis=1) 
            if float(resol)/1000 > fxmax:
                fxmax = float(resol)/1000  # Maximum of the observations
                imaxf = num_observations-1  
            f.write( '{0:d} {1:.6f} {2:.6f} {3:.6f} {4:.6f} {5:.6f} {6:.6f} {7:.6f} {8:.6f} {9:.3f} {10:.4f} {11:d}\n' .format(i, q1s, q2s, q3s, q4s, q5s, q6s, q7s, float(resol), fxmax, newdistPoints, eliminate) )
            f.close()
            # Save observation matrices
            np.savetxt('observations_x_observed.txt', x_observed)
            np.savetxt('observations_f_observed.txt', f_observed)
            np.savetxt('observations_theta.txt', theta)
	
        f_observed = np.transpose(f_observed)  # transform to column 
        theta =      np.transpose(theta)  # transform to column 
        # Finding the adaptive covariance matrix and transforming/reducing the matrix of observations
        x_observed, f_observed, theta, KK, eliminate = reduce(x_observed, f_observed, theta, imaxf) 
        KInv = np.linalg.inv(KK)   # Inverse of covariance matrix
        print('time (sec) after covariance: %f' % (timeit.default_timer() - startTime))  

        # Removing sampling file from previous step
        cmd = 'rm -f temp-sampling.txt'
        failure, output = commands.getstatusoutput(cmd)
        try: 
            pool = multiprocessing.Pool()  # Take as many processes as possible			
        except: 
            for c in multiprocessing.active_children():
                os.kill(c.pid, signal.SIGKILL)
                pool = multiprocessing.Pool(1)  # Number of processes to be used during PI sampling	
        for j in range(0, int(num_points/250)):
            pool.apply_async( samplePS, [250,fxmax] )
        pool.close()
        pool.join()
        reader = np.asmatrix(np.loadtxt('temp-sampling.txt'))
        x = np.transpose(np.asmatrix( reader[ np.argmax([ x[:,7] for x in reader] ), [0,1,2,3,4,5,6]] ))
        print('time (sec) after sampling: %f' % (timeit.default_timer() - startTime))  
	
        x_observed = np.hstack((x_observed,x))  # Adding next point to try               
        newdistPoints = 0
        num_observations = np.shape(x_observed)[1]
        if num_observations > 1:
            newdistPoints = distPoints(x_observed, num_observations-1, num_observations-2)/(theta[num_observations-2]+theta_nom) # calculating new distance between points
            newdistPoints = newdistPoints[0,0]
        f_observed = np.transpose(f_observed)  # transform to row 
        theta =      np.transpose(theta)  # transform to row 

print ('Final time (sec): %f' % (timeit.default_timer() - startTime))

#evolution( x_observed[:,0:-1], f_observed, startN+num_steps ) # Plots and saves resolution and magnetic fields as a function of iteration

