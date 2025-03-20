import h5py
import utils
import numpy as np
import pickle
import cv2
import os

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

def proofread_singlevid(vidpath, infdict):
    # infdict = load_infoutput(inf_dict)
    print(infdict['tracks'].shape)
    #keypoints: [b'Nose' b'TTI' b'Head' b'Trunk' b'Neck']
    cap = cv2.VideoCapture(vidpath)
    frame_count = 0 
    tracks = infdict['tracks']
    max_frames = tracks.shape[-1]
    numtracks = tracks.shape[0]
    instance_scores = infdict['instance_scores']
    mask_value_mat = np.zeros(instance_scores.shape)
    mask_value_mat[:] = np.nan
    dists_used = np.zeros(instance_scores.shape)
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret: # Break the loop if there are no more frames
                break
            image = frame[:,:,0]
            for track_id in range(numtracks):
                pose = tracks[track_id,:,:,frame_count]
                tti = pose[:,1]
                trunk = pose[:,3]
                if not np.isnan(trunk).any(): #trunk is present
                    dist_use = 30
                    if not np.isnan(tti).any():
                        dist_use = np.linalg.norm(trunk-tti)
                    # print(f"dist used: {dist_use}")
                    sample_points = utils.create_bounding_box(trunk[0], trunk[1], dist_use)
                    mask_value, mask = utils.mask_wrapper(image, sample_points, plot=False)
                    # print(f'MASK VALUE: {mask_value}')
                    mask_value_mat[track_id, frame_count] = mask_value
                    dists_used[track_id, frame_count] = dist_use
            frame_count += 1
    cap.release()
    return mask_value_mat, dists_used

def proofreading_wrapper(infs_path, vids_path):
    #loads in two lists that map to each other (index wise)
    inf_files, vid_files = utils.prep_files(infs_path,vids_path)
    iternum = 0
    for inf, vid in zip(inf_files, vid_files):
        print(f"ITERATION: {iternum}")
        print(inf)
        print(vid)
        inf_dict = load_infoutput(infs_path + inf)
        mvm, du = proofread_singlevid(vids_path+vid, inf_dict)
        save_path_mvm = "Z:/Charlie/charlie_data/pascal_videos/mvm/"+inf.split('.')[0]+'_mvm.pkl'
        save_path_du = "Z:/Charlie/charlie_data/pascal_videos/du/" + inf.split('.')[0]+'_du.pkl'
        with open(save_path_mvm, 'wb') as handle:
            pickle.dump(mvm, handle)
        with open(save_path_du, 'wb') as handle2:
            pickle.dump(du, handle2)
        iternum += 1
    return True

def main():
    videos_path = "Z:/Charlie/charlie_data/pascal_videos/805_819_vids/"
    infs_path = "Z:/Charlie/charlie_data/pascal_videos/infoutputs_norepeats/"
    completion_status = proofreading_wrapper(infs_path, videos_path)
    print(completion_status)
if __name__ == '__main__':
    main()