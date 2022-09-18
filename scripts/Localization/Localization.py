#!/usr/bin/env python3

"""
Created on Tue April 26 2022
@author: Nikhil Raj & Ronak Satpute

This is the main Localization Class.
Description: This class contains methods used for computation of 
             spatial position of traffic light.

"""

import numpy as np
import cv2

from final_global_var import *


class Localization():
    '''
    The Localization object takes in combination of LIDAR data and
    RGB Image and projection matrix to compute 3D Spatial Coordinates
    of the traffic lights. 

    This is the main computation logic.

    Parameters:
        point_cloud_data (numpy.ndarray) : Raw Point Cloud Data in shape
                                           (n x 3)
        rgb_image (numpy.ndarray) : Raw Image Data in shape (Width x Height
                                    x 3)
        projection_matrix (numpy.ndarray) : Projection Matrix in shape 
                                            (1 x 12)
        
    Returns::
    
    '''

    def __init__(self, point_cloud_data, rgb_image, projection_matrix):     
        self.raw_pcd = point_cloud_data   
        self.raw_rgb_image = rgb_image
        self.proj_mat = projection_matrix

        # Projected PCD after truncation 
        self.proj_pcd = None
        self.truncated_raw_pcd = None

    # Truncating the proj_pcd and truncated raw data to image plane size
    def proj_pcd_to_image_size(self, proj_pcd, truncated_pcd):
        '''
        Removes the projected data which lies outside the canvas of Image.
        Stores them in class variables.

        Parameters:
            proj_pcd (numpy.ndarray) : Projected Point Cloud Data of shape 
                                       (n x 2)
            truncated_pcd (numpy.ndarray) : Truncated Raw PCD Data of shape
                                       (n x 3)

        Returns:

        '''

        # Truncating u between 0 and 1920(Width)
        idx_u_low = np.asarray(np.where(proj_pcd[:, 0] < 0))
        mod_proj_pcd = np.delete(proj_pcd, idx_u_low, 0)
        mod_truncated_pcd = np.delete(truncated_pcd, idx_u_low, 0)

        idx_u_high = np.asarray(np.where(mod_proj_pcd[:, 0] > WIDTH))
        mod_proj_pcd = np.delete(mod_proj_pcd, idx_u_high, 0)
        mod_truncated_pcd = np.delete(mod_truncated_pcd, idx_u_high, 0)

        # Truncating u between 0 and 1080(Height)
        idx_v_low = np.asarray(np.where(mod_proj_pcd[:, 1] < 0))
        mod_proj_pcd = np.delete(mod_proj_pcd, idx_v_low, 0)
        mod_truncated_pcd = np.delete(mod_truncated_pcd, idx_v_low, 0)

        idx_v_high = np.asarray(np.where(mod_proj_pcd[:, 1] > HEIGHT))
        mod_proj_pcd = np.delete(mod_proj_pcd, idx_v_high, 0)
        mod_truncated_pcd = np.delete(mod_truncated_pcd, idx_v_high, 0)

        # Assigning results to class variables
        self.proj_pcd = mod_proj_pcd
        self.truncated_raw_pcd = mod_truncated_pcd


    # Projecting Data to 2D after Truncating the data to keep only points that
    # are in front of the vehicle. 
    def project_point_cloud(self):
        '''
        Projects 3D Point cloud data to 2D i.e. x, y, z (spatial 
        coordinates) converted to u, v (pixel coordinates). 

        Parameters:
        
        Returns:

        '''

        # Index of all the negative y values
        idx = np.asarray(np.where(self.raw_pcd[:, 2] < 0))

        # Deleting theses values from point cloud
        truncated_pcd = np.delete(self.raw_pcd, idx, 0)

        # Initialize empty list for storing projected data.
        proj_pcd = [] 

        # Iterating over rows of input PCD.
        for row in truncated_pcd:
            # Converting to homogenised coordinate system e.g. [X Y Z 1]
            li = list(row)
            li.append(1)
            hom_coord = np.asarray(li)
            hom_coord = hom_coord.reshape(4,1)

            # Projecting each row to image plane. 
            proj_2d = np.dot(self.proj_mat, hom_coord)
            proj_coord = [proj_2d[0]/proj_2d[2], proj_2d[1]/proj_2d[2]]
            proj_pcd.append(proj_coord)

        # Converting projected list to array of shape [lenngth of list, 2]
        proj_pcd = np.asarray(proj_pcd) 
        proj_pcd = proj_pcd.reshape(proj_pcd.shape[0], 2)

        # Converting array datatype from float to int.
        proj_pcd = proj_pcd.astype(int) 

        self.proj_pcd_to_image_size(proj_pcd, truncated_pcd)


    # Find the pixels within the contour
    def find_contour_pixels(self, contour):
        '''
        Find the pixel coordinates of the area enclosed by the contour

        Parameters:
            contour (numpy.ndarray) : Contour boundary data of shape 
                                      (n x 1 x 2)
        Returns:
            pixelpoints (numpy.ndarray) : Enclosed are coordinates of shape 
                                          (n x 2)

        '''

        contour = contour.reshape(contour.shape[0], 2)
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        image = cv2.drawContours(mask,[contour],0, 255,-1)
        pixelpoints = np.transpose(np.nonzero(mask))
        
        # Swapping Columns for homogeniety
        pixelpoints[:, [1, 0]] = pixelpoints[:, [0, 1]]

        return pixelpoints


    # 2D - Column Sort and check
    def col2Dsort(self, data):
        '''
        Sorts 2D array of size (n x 2), along first column first and then
        along second column.

        Parameters:
            data (numpy.ndarray) : Data to be sorted
                                      (n x 2)
        Returns:
            sorted_data (numpy.ndarray) : Sorted Data
                                          (n x 2)

        '''
        new_list = []

        # Sorting over first column in ascending order
        sorted_first_col = data[data[:,0].argsort()]

        uniques = np.unique(sorted_first_col[:, 0])

        # Sorting over second column in ascending order
        for item in uniques:
            idx = np.argwhere(sorted_first_col[:, 0] == item)

            if len(idx) == 1:
                new_list.append(sorted_first_col[idx].reshape(1,2))

            elif len(idx) > 1:
                temp_li = []

                for i in range(len(idx)):
                    temp = sorted_first_col[idx[i], :]
                    temp_li.append(temp)
                y_arr = np.asarray(temp_li).reshape(len(temp_li), 2)
                sortedArr_y = y_arr[y_arr[:,1].argsort()]

                for element in sortedArr_y:
                    new_list.append(element.reshape(1, 2))
        
        # Converting List to array and reshaping the array for homogeniety
        sorted_data = np.asarray(new_list)
        sorted_data = sorted_data.reshape(sorted_data.shape[0], 2)
        
        return sorted_data


    # Region of Interest in both Image Canvas and Projected LIDAR data
    def find_ROI(self, img_pixels, point_cloud):
        '''
        Find region of interest where the search for intersection has to
        be performed. This is done to reduce the search space. 

        Parameters:
            img_pixels (numpy.ndarray) : Image data
                                         (n x 2)
        Returns:
            point_cloud (numpy.ndarray) : Point cloud data
                                          (n x 2)

        '''

        # Finding the maximum and minimum value in image in order to truncate
        # the point cloud data which lies outside the bounds.
        img_pixels_max_val = np.max(img_pixels, axis = 0)
        img_pixels_min_val = np.min(img_pixels, axis = 0)

        u_limit_low = np.asarray(np.where(point_cloud[:, 0] 
                                 < img_pixels_min_val[0]))
        mod_point_cloud = np.delete(point_cloud, u_limit_low, 0)

        u_limit_high = np.asarray(np.where(mod_point_cloud[:, 0] 
                                           > img_pixels_max_val[0]))
        mod_point_cloud = np.delete(mod_point_cloud, u_limit_high, 0)


        # If the resulting point cloud is not empty then maximum and minimum 
        # of point cloud is determined to truncate the image which lies 
        # outside the bounds
        if mod_point_cloud.size != 0:
            pcMaxVal = np.max(mod_point_cloud, axis = 0)
            pcMinVal = np.min(mod_point_cloud, axis = 0)

            pc_limit_low = np.asarray(np.where(img_pixels[:, 1] < pcMinVal[1]))
            mod_img_pixels = np.delete(img_pixels, pc_limit_low, 0)

            pc_limit_high = np.asarray(np.where(mod_img_pixels[:, 1] 
                                                > pcMaxVal[1]))
            mod_img_pixels = np.delete(mod_img_pixels, pc_limit_high, 0)


            return mod_point_cloud, mod_img_pixels
        
        else:

            return mod_point_cloud, img_pixels


    # Finding the threshold for each value i.e. upper bound and lower bound 
    # for the purpose of finding the overlapping points in both image and 
    # LIDAR point
    def crit_tol_limit(self, val):
        '''
        Based on the accepted tolerance constants the threshold for each 
        coordinate is determined.

        Parameters:
            Val (int) : Value whose threshold are to be determined.
        Returns:
            lower_bound (int) : Lower bound value.
            upper_bound (int) : Upper bound value.

        '''
        threshold = (REL_TOL * abs(val)) if (REL_TOL * abs(val)) >= MAX_TOL \
                                         else MAX_TOL
        lower_bound = val - threshold
        upper_bound = val + threshold

        return lower_bound, upper_bound


    # Region of Interest in both Image Canvas and Projected LIDAR data
    def intersection_pc_im(self, imgPixels, pointCloud):
        '''
        Finding the intersection between the image white pixel coordinate 
        and LIDAR point cloud based on the computed region of interest and 
        critical threshold values.

        Parameters:
            img_pixels (numpy.ndarray) : Image data
                                         (n x 2)
            pointCloud (numpy.ndarray) : Point cloud data
                                         (n x 2)
        Returns:

        '''
        small_loop_counter = 0
        total_loop_counter = 0
        break_counter = 0

        intersection_list = []

        # ROI Point Cloud
        mod_pointCloud, mod_imgPixels = self.find_ROI(imgPixels, pointCloud)

        if mod_pointCloud.shape[0] >= imgPixels.shape[0]:
            big_arr = mod_pointCloud
            small_arr = mod_imgPixels
            append_flag = 'big'
        else:
            big_arr = mod_imgPixels
            small_arr = mod_pointCloud
            append_flag = 'small'

        max_counter = big_arr.shape[0]

        for i in range(small_arr.shape[0]):

            big_loop_counter = 0

            u_val = small_arr[i, 0]
            v_val = small_arr[i, 1]
            u_lower_bound, u_upper_bound = self.crit_tol_limit(u_val)
            v_lower_bound, v_upper_bound = self.crit_tol_limit(v_val)

            # canvas limit
            if u_lower_bound <= 0:
                u_lower_bound = 0
            elif u_upper_bound >= WIDTH:
                u_upper_bound = WIDTH
            elif v_lower_bound <= 0:
                v_lower_bound = 0
            elif v_upper_bound >= HEIGHT:
                v_upper_bound = HEIGHT


            while not big_loop_counter >= max_counter:

                if (u_upper_bound > big_arr[big_loop_counter, 0]):
                    if ((big_arr[big_loop_counter, 0] <= u_upper_bound) and (big_arr[big_loop_counter, 0] >= u_lower_bound)):
                        if ((big_arr[big_loop_counter, 1] <= v_upper_bound) and (big_arr[big_loop_counter, 1] >= v_lower_bound)):

                            if append_flag == 'small':
                                intersection_list.append(small_arr[i])
                            elif append_flag == 'big':
                                intersection_list.append(big_arr[big_loop_counter])
                            else: 
                                print('Append Flag Error')
                    big_loop_counter += 1
                else:
                    break_counter += 1
                    break


                big_loop_counter += 1 

            total_loop_counter += big_loop_counter

            small_loop_counter += 1  

        total_loop_counter = total_loop_counter


        intersection_list = list(set(tuple(x) for x in intersection_list)) 

        intersection_array = np.asarray(intersection_list)


        return intersection_array

    # Reverse Search
    def reverse_search(self, intersection_array, projected_pcd, 
                       truncated_pcd):
        '''
        Using the idea of backtracking the mean, maximum and minimum
        value are computed in spatial coordinate system for the intersection
        points found.

        Parameters:
            intersection_array (numpy.ndarray) : Computed intersection data
                                                 (n x 2)
            projected_pcd (numpy.ndarray) : Projected Point cloud data
                                                 (n x 2)
            truncated_pcd (numpy.ndarray) : Raw Point cloud data
                                                 (n x 3)
        Returns:
            result (numpy.ndarray) : Spatial coordinate information
                                                 (1 x 9)

        '''

        # Converting all the arrays to list for backtracking
        intersection_li = list(list(x) for x in intersection_array.tolist())
        pcd_unsorted_li = list(list(x) for x in projected_pcd.tolist())
        truncated_pcd_li = list(list(x) for x in truncated_pcd.tolist())

        # Creating empty list for storing the values
        sorted_li, result_li = [], []

        # Itering over each list for finding the corresponding spatial 
        # coordinates
        for element in intersection_li:
            sorted_li.append(pcd_unsorted_li.index(element))

        for count in range(len(sorted_li)):
            result_li.append(truncated_pcd_li[count])

        result_arr = np.asarray(result_li)

        # Checking whether the result array is empty or not, after which 
        # final result list is populated accordingly
        if result_arr.size != 0:
            min_vals = np.min(result_arr, axis = 0)
            mean_vals = np.mean(result_arr, axis = 0)
            max_vals = np.max(result_arr, axis = 0)

            result = np.asarray([mean_vals[0], max_vals[0], min_vals[0],
                                 mean_vals[1], max_vals[1], min_vals[1],
                                 mean_vals[2], max_vals[2], min_vals[2]])

            return result
        else:
            result = IGNORED_DEFAULT_VALUES
            return result



    def localization_driver(self):
        '''
        Driver code which utilizes all the above mentioned class methods 
        to perform the computation for the set of image and LIDAR point 
        cloud information.

        Parameters:

        Returns:
            final_result (list) : Spatial coordinate information of all the 
                                  detected traffic light in the segmented mask
                                  (no of traffic light detected x 1 x 9)


        '''

        # Project 3D Point Cloud to 2D and truncate accordingly
        # to reduce search space
        self.project_point_cloud()

        sorted_projected_point_cloud = self.col2Dsort(self.proj_pcd)
        
        # Finding contours within the binary segmented mask
        # GrayScale, Binary threshold & contour finding
        image_gray = cv2.cvtColor(self.raw_rgb_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(image = thresh, mode = cv2.RETR_TREE, 
                                               method = cv2.CHAIN_APPROX_NONE)

        final_result = []
        for contour_idx in range(len(contours)):
            pixelpoints = self.find_contour_pixels(contours[contour_idx])

            if cv2.contourArea(contours[contour_idx]) <= AREA_THRESHOLD:
                final_result.append(IGNORED_DEFAULT_VALUES)
            else:
               
                sorted_pixelpoints = self.col2Dsort(pixelpoints)

                intersection_array = self.intersection_pc_im(sorted_pixelpoints,
                                     sorted_projected_point_cloud)

                result = self.reverse_search(intersection_array, self.proj_pcd, 
                                             self.truncated_raw_pcd)
                final_result.append(result)
        

        return(final_result)
