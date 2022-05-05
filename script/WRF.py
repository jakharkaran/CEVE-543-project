import matplotlib.pyplot as plt
import netCDF4 as nc                      # Library to read .netcdf files
import findiff                            # Library to calculate spacial derivatives
from derivative_WRF import derivative_WRF
from itertools import combinations
import gc

import sys
import numpy as np
import scipy.io as sio
import math
import os

from eqsdiscovery.rvm import RVR
from eqsdiscovery.rvm import RVR
from eqsdiscovery.denormalizeWeight import denormalizeWeight
from eqsdiscovery.util import corr2
from eqsdicovery.WRF_library import subsampled_linear_derivative_library

system = 'WRF'
Nfilter = '1000' # 1000 250 
libraryDerivativeOrder = 3
thresholdAlpha = 50

#### subsampling factor = 1 will subsample all data
#### subsampling factor = 0 will subsample no data
subsampling_factor = .000001

#### Maximum order of derivatives for non-linear terms
#### UxUxx : order=3, UxVxxWx : order=4
non_linear_order = 4

#### Degree of non linear terms [>= 1]
#### Tx:1, TxTx:2, TxxTxTxxx:3, ..
non_linear_term_polynomial_degree = 1

yName = 'R13'           # Reynold's Stress
filterType = 'gaussian' # Gaussian Filter
N = '1024'              # No of grid points in zonal and meridional direction

tolerance = 1e-6
nosSnapshot = 1
skipSnapshot = 1
subtractModel = '0'
SAVE_DIR = './'
dealias = 0
galileanInvariance = True
Delta = int(Nfilter)
RESULT_DIR = SAVE_DIR

DATA_DIR = '../data/' + system  + '/' + Nfilter + 'km/'

# Maximum order of spatial derivatives to be included in the library of basis functions
order_max = np.array([1, 1, 1])*libraryDerivativeOrder 
order = order_max

h = [3, 3, .5]                        # Distance between adjacent grid points [in km]

#### Importing data 

if Nfilter == '1000':
    # Filter size = 1000 km
    filename_theta = DATA_DIR + 'lp1000_eq_dis_wrfout_d01_2016-08-04_00:00:00.nc'     # Wind velocities
    filename_y = DATA_DIR + 'uw_lp1000_spec_eq_dis_wrfout_d01_2016-08-04_00:00:00.nc' # Reynold's Stress
            
if Nfilter == '250':
    # Filter size = 250 km
    filename_y = DATA_DIR + 'uw_lp250_spec_eq_dis_wrfout_d01_2016-08-04_00:00:00.nc'
    filename_theta = DATA_DIR + 'lp250_lp250_eq_dis_wrfout_d01_2016-08-04_00:00:00.nc'
    
r_dataset = nc.Dataset(filename_y)
uvw_dataset = nc.Dataset(filename_theta)

print(r_dataset.variables.keys()) # name of all variables
print(uvw_dataset.variables.keys()) # name of all variables

r_data = r_dataset.variables['tau_rey']
u_data = uvw_dataset.variables['u_lev']
v_data = uvw_dataset.variables['v_lev']
w_data = uvw_dataset.variables['w_lev']

y = np.transpose(r_data[:])
f1 = np.transpose(u_data[:])
f2 = np.transpose(w_data[:])

del r_dataset, uvw_dataset, r_data, u_data, v_data, w_data
gc.collect()

#########################

libraryTerms = ['U', 'W']        # Wind velocities included in the library
yname = 'R13'

# Forming the derivative library

#### Boundary points
## Remove boundary points since the boudary point derivatives are calculated using numerical scheme of lesser order, increasing the error in derivative calculation nimerically
remove_boundary_points = True
N_bd_points = 4

#### Numerical order of accuracy of derivatives
acc_order = 2

print('subsampled gridpoints', 1016*1016*92*subsampling_factor)

############################

# subsampling random indices
if remove_boundary_points:
    f1shape = np.asarray(np.shape(f1)) - 2*N_bd_points
    y_shape = y.shape
    y_bd = y[N_bd_points:y_shape[0]-N_bd_points,N_bd_points:y_shape[1]-N_bd_points,N_bd_points:y_shape[2]-N_bd_points]

else:
    f1shape = np.asarray(np.shape(f1))
    y_bd = y

subsample_size = int(np.prod(f1shape)*subsampling_factor)
subsample_ind = np.random.randint(low=0, high=np.prod(f1shape), size=subsample_size)

y_flat = np.ravel(y_bd)
y_subsampled = y_flat[subsample_ind]

del y_bd, y
gc.collect()

print('subsampled y')
    
#### Calculating linear derivatives of each of the linrary terms (wind velocities)    

if len(libraryTerms) >= 1:
    
    f1_derivative_linear_subsampled_arr, f1name_derivative_linear_subsampled_arr = subsampled_linear_derivative_library(
        f1, fname=libraryTerms[0], h=h, order_max=order, acc_order=acc_order, remove_boundary_points=remove_boundary_points, N_bd_points=N_bd_points, subsampling_factor=subsampling_factor,subsample_ind=subsample_ind)
    
if len(libraryTerms) >= 2:    
    f2_derivative_linear_subsampled_arr, f2name_derivative_linear_subsampled_arr = subsampled_linear_derivative_library(
        f2, fname=libraryTerms[1], h=h, order_max=order, acc_order=acc_order, remove_boundary_points=remove_boundary_points, N_bd_points=N_bd_points, subsampling_factor=subsampling_factor,subsample_ind=subsample_ind)

if len(libraryTerms) >= 3:   
    f3_derivative_linear_subsampled_arr, f3name_derivative_linear_subsampled_arr = subsampled_linear_derivative_library(
        f3, fname=libraryTerms[2], h=h, order_max=order, acc_order=acc_order, remove_boundary_points=remove_boundary_points, N_bd_points=N_bd_points, subsampling_factor=subsampling_factor,subsample_ind=subsample_ind)

print('calculated f1/f2/f3 derivatives and subsampled')    

if len(libraryTerms) == 1:
    f_derivative_linear_subsampled_arr = f1_derivative_linear_subsampled_arr
    fname_derivative_linear_subsampled_arr = fname_derivative_linear_subsampled_arr
    
    del f1_derivative_linear_subsampled_arr
    
if len(libraryTerms) == 2:
    f_derivative_linear_subsampled_arr = np.concatenate(
        (f1_derivative_linear_subsampled_arr, f2_derivative_linear_subsampled_arr), axis=0)
    fname_derivative_linear_subsampled_arr = np.concatenate(
        (f1name_derivative_linear_subsampled_arr,f2name_derivative_linear_subsampled_arr), axis=0)
    
    del f1_derivative_linear_subsampled_arr, f2_derivative_linear_subsampled_arr

if len(libraryTerms) == 3:
    f_derivative_linear_subsampled_arr = np.concatenate(
        (f1_derivative_linear_subsampled_arr, f2_derivative_linear_subsampled_arr, f3_derivative_linear_subsampled_arr), axis=0)
    fname_derivative_linear_subsampled_arr = np.concatenate(
        (f1name_derivative_linear_subsampled_arr,f2name_derivative_linear_subsampled_arr, f3name_derivative_linear_subsampled_arr), axis=0)
    
    del f1_derivative_linear_subsampled_arr, f2_derivative_linear_subsampled_arr, f3_derivative_linear_subsampled_arr

print('Move f1/f2/f3 derivatives in a single array') 

### Forming non-linear library

# Indices for non linear terms

# creating array of all indices of linear elements
index_list = np.linspace(
    0,len(fname_derivative_linear_subsampled_arr)-1,len(fname_derivative_linear_subsampled_arr))
## converting float elements to integer
index_list = index_list.astype(int)  

## calculating possible combination for all the indices
temp_comb = combinations(index_list,2)
index_list_comb = [r for r in temp_comb]
index_list_comb = np.asarray(index_list_comb)
index_list_comb = index_list_comb.astype(int)

#### Forming the linear with nonlinear terms, 

for ind in index_list_comb:
    
    temp_fname = fname_derivative_linear_subsampled_arr[ind[0]] + fname_derivative_linear_subsampled_arr[ind[1]]
    
    if len(temp_fname) <= non_linear_order + 2:
        # Excluding terms with sum of order of derivatives of all terms > non_linear_order
        fname_derivative_linear_subsampled_arr = np.append(
            fname_derivative_linear_subsampled_arr,np.array(temp_fname))

        temp_f = np.multiply(f_derivative_linear_subsampled_arr[ind[0]] , f_derivative_linear_subsampled_arr[ind[1]])
        temp_f = np.reshape(temp_f,[1,len(temp_f)])
        f_derivative_linear_subsampled_arr = np.append(
            f_derivative_linear_subsampled_arr,temp_f,axis=0)

theta = f_derivative_linear_subsampled_arr.T   # Flattened library of basis function
thetaAllName = fname_derivative_linear_subsampled_arr

del f_derivative_linear_subsampled_arr
gc.collect()

############################
    
print('------------------------------------------------------------')
print('filterType                    =  ',filterType)
print('yName                         =  ',yName)
print('Library Terms                 =  ',libraryTerms)
print('Nfilter                       =  ',Nfilter)
print('libraryDerivativeOrder        =  ',libraryDerivativeOrder)
print('Threshold Alpha               =  ',thresholdAlpha)
print('Tolerance                     =  ',tolerance)
print('libraryDerivativeOrder        =  ',libraryDerivativeOrder)
print('Snapshots                     =  ',nosSnapshot)
print('skipSnapshot                  =  ',skipSnapshot)
print('Subtract Model                =  ',subtractModel)
print('dealiased data                =  ',dealias)
print('Filter Size                   =  ',Delta)
print('Data Directory                =  ',DATA_DIR)
print('Result Directory              =  ',RESULT_DIR)
print('------------------------------------------------------------')

# .mat files are column ordered while python files are row ordered
# Take a transpose 

print('Library Formed of # ' + str(len(thetaAllName)) + ' terms')
print('Terms of the library thetaAllName = ')
print(thetaAllName)

# Normalizing input data 

thetaAllMean = theta.mean(axis=0)
thetaAllStd = theta.std(axis=0)

for count in range(0,theta.shape[1]):
    theta[:,count] = (theta[:,count]-thetaAllMean[count])/thetaAllStd[count]
    
print("Library of Functions Created\n")
print('Shape of Library (theta) =' , thetaAllMean.shape)

# y: Reynolds Stress

yUnNorm = y_subsampled
yMean = yUnNorm.mean(axis=0)
yStd = yUnNorm.std(axis=0)
y = (yUnNorm-yMean)/yStd
del yUnNorm, y_subsampled

print(y.shape)
print(theta.shape)
print(thetaAllName.shape)

#### """Sparse Bayesian Regression"""

print('----------------------------------------')
print("Starting Sparse Linear Regression on", yName)
print('Threshold Alpha    =  ',str(thresholdAlpha) )
print('Tolerance          =  ', tolerance)
print('========================================')

## List of outputs

clf = RVR(threshold_alpha= thresholdAlpha, tol=tolerance, verbose=True, standardise=True)
fitted = clf.fit(theta    , y     , thetaAllName        )
scoreR2 = clf.score_R2(theta,y) 
MSE     = clf.score_MSE(theta,y) 
weightMean  = clf.m_#[0]
alpha   = clf.alpha_

i_s = clf.beta_ * np.dot( clf.phi.T, clf.phi ) +  np.diag( clf.alpha_ )
sigma_ = np.linalg.inv(i_s)
weightStd = np.sqrt(np.diag(sigma_))

retainedInd = np.where(clf.retained_)[0]
thetaName = [thetaAllName[v] for v in retainedInd]
thetaMean = np.array([thetaAllMean[v] for v in retainedInd])
thetaStd = np.array([thetaAllStd[v] for v in retainedInd])

errorBar = np.zeros(len(sigma_), dtype=float)
for count in range(0,len(sigma_)):
    errorBar[count] = (weightStd[count]/(weightMean[count]))**2

# De-normalizing data / Scaling discovered weights

weightMeanScaled, constant = denormalizeWeight(weightMean, yMean, yStd, thetaMean, thetaStd)

weightMeanScaledDelta = np.zeros(len(thetaName))
for count in range(0,len(thetaName)):
    tempU = str.count(thetaName[count],'U')
    tempV = str.count(thetaName[count],'V')
    tempW = str.count(thetaName[count],'W')

    orderTerm = len(thetaName[count]) - tempU - tempV - tempW
    print(orderTerm,tempU,tempV,tempW,thetaName[count])

    weightMeanScaledDelta[count] = weightMeanScaled[count]/(Delta**orderTerm)

print('------------------------------------------------------------')
print('Threshold Alpha        =  ', str(thresholdAlpha) )
print('Tolerance              =  ', tolerance)
print('Bases Retained         =  ', clf.n_retained       )
print('Model R2 score         =  ', clf.score_R2(theta,y)  )
print('Model MSE              =  ', clf.score_MSE(theta,y) )
print('Error Bar              =  ', np.sum(errorBar))
print('------------------------------------------------------------')
print('Weights Normalized     =  ', weightMean)
print('Std of bases           =  ', weightStd)
print('Error Bar Total        =  ', errorBar)
print('============================================================')
print('Weights Scaled         =  ', weightMeanScaled)
print('Delta                  =  ', Delta)
print('Weights Scaled (Delta) =  ', weightMeanScaledDelta)
print('============================================================')

#### Naming file to be saved with data

tempstr = yName + "".join(libraryTerms)
tempstr2 = '_' + 'NX' + '1024'

tempstr3 = '_' + str(int(Nfilter))

tempstr4 = '_O' + str(int(libraryDerivativeOrder))
tempstr5 = '_T' + str(int(nosSnapshot))

if (dealias == 1):
    tempstr6 = '_dealias'
else:
    tempstr6 = ''

if (subtractModel == '0'):
    tempstr7 = ''
else:
    tempstr7 = '_' + subtractModel
    
tempstr8 = '_accO' + str(acc_order) 
tempstr9 = '_sample' + str(subsampling_factor)
tempstr10 = '_NLterm' + str(non_linear_term_polynomial_degree)
tempstr11 = '_NLO' + str(non_linear_order)

filename = tempstr + '_' + filterType + tempstr2 + tempstr3 + tempstr4 + '_' + str(int(thresholdAlpha)) + '_' + str(tolerance) + tempstr5 + tempstr6 + tempstr7 + tempstr8 + tempstr9 + tempstr10 + tempstr11

try:
    os.mkdir(RESULT_DIR)
except OSError as error:
    print(error)
    
# Saving data

mdict = {'filterType':filterType,
         'thresholdAlpha':thresholdAlpha, 'tolerance':tolerance, 'Nfilter':Nfilter,
         'yName':yName, 'yMean':yMean, 'yStd':yStd, 'thetaMean':thetaMean,
         'thetaStd':thetaStd, 'weightMean':weightMean, 'weightStd':weightStd,
         'weightMeanScaled':weightMeanScaled, 'constant':constant,
         'weightMeanScaledDelta':weightMeanScaledDelta, 'Delta':Delta,
         'errorBar':errorBar, 'MSE':MSE, 'alpha':alpha, 'R2':scoreR2,
         'thetaName':thetaName, 'thetaAllName':thetaAllName, 'skipSnapshot':skipSnapshot,
         'libraryDerivativeOrder':libraryDerivativeOrder, 'libraryTerms':libraryTerms, 'noOfBases':clf.n_retained,
         'galileanInvariance':galileanInvariance, 'subtractModel':subtractModel, 'dealias':dealias,
         'retainedInd':retainedInd, 'nosSnapshot':nosSnapshot}
sio.savemat(RESULT_DIR+filename+'.mat', mdict)
# To load .mat file use sio.loadmat(file_name.mat, simplify_cells=True)
# Each term thetaAllName and other strings would be saved as characters of equal length. Empty spaces would be used to make all the characters of equal lengths. Use the following to remove the empty space in these.
# thetaAllName = [word.strip() for word in thetaAllName]

np.savez(RESULT_DIR+filename , filterType=filterType,
         thresholdAlpha=thresholdAlpha, tolerance=tolerance, Nfilter=Nfilter,
         yName=yName, yMean=yMean, yStd=yStd, thetaMean=thetaMean,
         thetaStd=thetaStd, weightMean=weightMean, weightStd=weightStd,
         weightMeanScaled=weightMeanScaled, constant=constant,
         weightMeanScaledDelta=weightMeanScaledDelta, Delta=Delta,
         errorBar=errorBar, MSE=MSE, alpha=alpha, R2=scoreR2,
         thetaName = thetaName, thetaAllName=thetaAllName,
         libraryDerivativeOrder = libraryDerivativeOrder, libraryTerms=libraryTerms, noOfBases=clf.n_retained,
         galileanInvariance=galileanInvariance, subtractModel=subtractModel, dealias=dealias,
         retainedInd=retainedInd, nosSnapshot=nosSnapshot)
print(filename)

###############################################
