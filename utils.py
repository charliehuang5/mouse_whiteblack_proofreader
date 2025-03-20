import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os

def create_bounding_box(x, y, L):
    half_length = L // 2
    # Create arrays for x and y coordinates
    maxheight = 1023
    maxwidth = 1279
    x_coords = np.arange(x - half_length, x + half_length + 1)
    x_coords = np.minimum(x_coords, maxwidth)
    y_coords = np.arange(y - half_length, y + half_length + 1)
    y_coords = np.minimum(x_coords, maxheight)
    
    # Create a meshgrid of coordinates
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    
    # Stack and reshape the coordinates into a list of integer coordinates
    coordinates = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    
    return coordinates

def create_mask(points, height=1024, width=1280):
    mask = np.empty([height,width])
    mask[:] = np.nan
    for i in range(points.shape[0]):
        mask[int(points[i,1]),int(points[i,0])] = 1
    return mask

def mask_wrapper(image, points, plot=False):
    mask = create_mask(points)
    within_mask = np.multiply(image, mask)
    mask_value = np.nanmean(within_mask)
    if plot:
        plt.figure(figsize=(20,10))
        plt.imshow(within_mask, cmap='gray', vmin=0, vmax=255)
    return mask_value, mask
import h5py
def load_infoutput(infpath):
    inf_dict = {}
    print('infpath: ', infpath)
    with h5py.File(infpath, 'r') as file:
        inf_dict['track_names'] = file['track_names'][:]
        inf_dict['tracks'] = file['tracks'][:]
        inf_dict['tracking_scores'] = file['tracking_scores'][:]
        inf_dict['instance_scores'] = file['instance_scores'][:]
        inf_dict['point_scores'] = file['point_scores'][:]
    return inf_dict

def prep_files(inf_files_path, video_files_path):
    vidfiles_sorted = []
    corresp_inf_files = []
    for inf_file in os.listdir(inf_files_path):
        iparts = inf_file.split('_')
        # print(iparts)
        mouse_id = iparts[0]
        date = iparts[1]
        epoch = iparts[2]
        des_parts = [mouse_id, date, epoch]
        vid_file = find_vid_file(video_files_path, des_parts)
        vidfiles_sorted.append(vid_file)
        corresp_inf_files.append(inf_file)
        if vid_file == "failed to find":
            print("vid file lacking for: ", inf_file)
            break
    return corresp_inf_files, vidfiles_sorted

def find_vid_file(video_files_path, des_part):
    for vid_file in os.listdir(video_files_path):
        date = vid_file.split('_')[0][5:]
        date = date.split('-')
        date = date[0]+date[1]
        # print(date)
        vparts = vid_file.split('-')
        vparts = vparts[-1].split('_')
        mouse_id = vparts[0]
        epoch = vparts[1].split('.')[0]
        # print(mouse_id, epoch)
        if des_part[0] == mouse_id and des_part[1] == date and des_part[2] == epoch:
            return vid_file
    return "failed to find"

def visualize_pose(video_path, frame_no, tracks_new, savepath=None):
    cap = cv2.VideoCapture(video_path)  # video_name is the video being called
    cap.set(1,frame_no)  # Where frame_no is the frame you want
    ret, frame = cap.read()  # Read the frame
    pose0new = tracks_new[0,:,:,frame_no]
    pose1new = tracks_new[1,:,:,frame_no]
    image = frame[:,:,0]
    fig, ax = plt.subplots(figsize=(20,10), sharey=True)
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax.scatter(pose0new[0,:], pose0new[1,:], c='r')
    ax.scatter(pose1new[0,:], pose1new[1,:], c='b')
    ax.set_title(f'New tracks, frame {frame_no}')
    if savepath == None:
        plt.show()
    else:
        plt.savefig(savepath)

#Unused

def generate_points_in_circle_gaussian(x, y, r, N):
    points = []
    for i in range(N):
        # Angle for the current point
        theta = 2 * np.pi * i / N
        # Radius for the current point (randomly selected from a Gaussian distribution)
        radius = np.abs(np.random.normal(loc=0, scale=r/3))  # Use absolute value to avoid negative radius
        if radius > r:  # Clamp radius to the maximum radius
            radius = r
        # Convert polar coordinates to Cartesian coordinates
        point_x = x + radius * np.cos(theta)
        point_y = y + radius * np.sin(theta)
        points.append([point_x, point_y])
    return np.array(points)