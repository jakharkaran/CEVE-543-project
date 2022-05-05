def subsampled_linear_derivative_library(
    f, fname, h, order_max, acc_order, remove_boundary_points, N_bd_points, subsampling_factor, subsample_ind):
#### Library of linear derivatives of f would be formed using finite difference with 'acc_order' of accuracy.
#### The library would contain derivatives of f upto order_max.
#### f: input 3D matrix.
#### fname: string with name of f
#### h: [hx, hy, hx]: distance between adjacent grid points 
#### order_max: [Ox, Oy, Oz]: Maximum order of linear derivatives 
#### acc_order: [even integer upto 10]: order of accuracy for the finite difference (upto 10)
#### remove_boundary_points: The boundary point derivatives will be calculated with lesser order of accuracy. Removing those booundary points
#### N_bd_points: Number of boundary points to be removed. 5 for O(10), 1 for O(1)
#### subsampling_factor: [inbetween 0 to 1]: fraction of grid points to be subsampled


    #### Array listing order of all liner derivatives upto order_max
    order_arr = [[x,y,z] for x in range(order_max[0]+1)  for y in range(order_max[1]+1) for z in range(order_max[2]+1)]

    # ####  Removing galeliann invariant term
    del order_arr[0]
    
    N_linear_terms = len(order_arr)

    #### Calculating derivatives of linear terms using finite differene 
    derivative_linear_arr = np.array([derivative_WRF(f,temp_order,h,acc_order,fname) for temp_order in order_arr])
    
    temp_derivative = derivative_linear_arr

    #### Removing boundary points, subsampling data

    count = 0;
    for x in derivative_linear_arr:
        temp = x[0]
        temp_shape = temp.shape
        
        #### Removing boundary points
        #### Boundary point have lesser order of accuracy of derivatives than the non-boundary points
        if remove_boundary_points:
            temp2 = temp[N_bd_points:temp_shape[0]-N_bd_points,N_bd_points:temp_shape[1]-N_bd_points,N_bd_points:temp_shape[2]-N_bd_points]
        else:
            temp2 = temp

        ## Flattening array
        temp_flatten = np.ravel(temp2)
#         temp_flatten_shape = np.shape(temp_flatten)[0]
        
#         print(np.shape(temp_flatten)[0])

#         ## size of subsampled data
#         temp_subsample_size = int(temp_flatten_shape*subsampling_factor)

#         # random index of elements to be subsampled
#         if count == 0:
#             temp_subsample_ind = np.random.randint(low=0, high=temp_flatten_shape, size=temp_subsample_size)

        #### Subsampling
        temp_subsampled = temp_flatten[subsample_ind]

        ## Initializing arrays
        if count == 0:
            derivative_linear_subsampled_arr = np.empty([derivative_linear_arr.shape[0],temp_subsampled.shape[0]])
            derivative_linear_subsampled_arr_fname = np.ndarray(N_linear_terms,dtype='O')

        ## Creating two different arrays for data and string referring to derivative of data
        derivative_linear_subsampled_arr[count,:] = temp_subsampled
        derivative_linear_subsampled_arr_fname[count] = x[1]
                
        count = count+1
                
    return derivative_linear_subsampled_arr, derivative_linear_subsampled_arr_fname


######################