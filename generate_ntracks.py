import utils
import pickle
import numpy as np
import matplotlib.pyplot as plt

def generate_nanmat(mvm_oi, guessed_wb_thresh = 90):
    nanmat = np.zeros([2,mvm_oi.shape[1]])
    nanmat[:] = np.nan
    whitemvm = np.zeros(mvm_oi.shape[1])
    whitemvm[:] = np.nan
    blackmvm = np.zeros(mvm_oi.shape[1])
    blackmvm[:] = np.nan

    print(nanmat.shape)
    nan_frames = []
    sing_frames = []
    for i in range(mvm_oi.shape[1]):
        # print(i)
        loc_nonan = np.where(~np.isnan(mvm_oi[:,i]))
        if len(loc_nonan[0]) == 0:
            nan_frames.append(i)
        if len(loc_nonan[0]) == 1:
            sing_frames.append(i)
            index = loc_nonan[0][0]
            val = mvm_oi[index, i]
            if val >= guessed_wb_thresh:
                nanmat[0,i] = index #row 0 is white
                whitemvm[i] = val
            else:
                nanmat[1,i] = index #row 1 is black
                blackmvm[i] = val
        elif len(loc_nonan[0]) == 2:
            index0 = loc_nonan[0][0]
            index1 = loc_nonan[0][1]
            mval0 = mvm_oi[index0, i]
            mval1 = mvm_oi[index1, i]
            whitetrack = index1
            blacktrack = index0
            if mval0 >= mval1:
                whitetrack = index0 
                blacktrack = index1
                whitemvm[i] = mval0
                blackmvm[i] = mval1
            else:
                whitemvm[i] = mval1
                blackmvm[i] = mval0
                

            nanmat[:,i] = [whitetrack, blacktrack] #row 0 is white, row 1 is black
    return nanmat, whitemvm, blackmvm

def generate_reorg_tracks(tracks, blackwhite_indsmat):
    copy_tracks = np.zeros(tracks.shape)
    copy_tracks[:] = np.nan
    for i in range(tracks.shape[-1]):
        whitetrack = blackwhite_indsmat[0,i]
        blacktrack = blackwhite_indsmat[1,i]
        if not np.isnan(whitetrack):
            # print(whitetrack)
            copy_tracks[0,:,:,i] = tracks[int(whitetrack),:,:,i]
        if not np.isnan(blacktrack):
            # print(blacktrack)
            copy_tracks[1,:,:,i] = tracks[int(blacktrack),:,:,i]        
    return copy_tracks

def main():
    mvm_path = "Z:/Charlie/charlie_data/pascal_videos/mvm/"
    videos_path = "Z:/Charlie/charlie_data/pascal_videos/805_819_vids/"
    infs_path = "Z:/Charlie/charlie_data/pascal_videos/infoutputs_norepeats/"
    save_path = "Z:/Charlie/charlie_data/pascal_videos/newtracks/"
    save_path_pics = "Z:/Charlie/charlie_data/pascal_videos/mvm_photos/"

    corresp_mvm_files, vidfiles_sorted = utils.prep_files(mvm_path,videos_path)
    mvm_vec_list = []
    whitevals = []
    blackvals = []
    for mvm_fname,vid_fname in zip(corresp_mvm_files, vidfiles_sorted):
        inf_file = mvm_fname[:-8] + '.analysis.h5'
        inf_dict = utils.load_infoutput(infs_path + inf_file)
        mvm_full = mvm_path+mvm_fname
        with open(mvm_full, "rb") as input_file:
            mvm = pickle.load(input_file)

        mvm_vec = np.hstack([mvm[0,:], mvm[1,:]])
        mvm_vec_list.append(mvm_vec)
        
        # plt.figure(figsize=(7,4))
        # plt.hist(mvm_vec)
        # plt.savefig(save_path_pics + mvm_fname[:-8] + '_mvmhist.png')   

        nanmat,whitemvm,blackmvm = generate_nanmat(mvm)
        whitevals.append(whitemvm)
        blackvals.append(blackmvm)
        break
        newtracks = generate_reorg_tracks(inf_dict['tracks'], nanmat)

        newtrack_fname = mvm_fname[:-8] + '_newtracks.pkl'
        print("new track generated, saving-- ", newtrack_fname)
        frames_query = np.arange(0,mvm.shape[-1],1000)
        for fno in frames_query:
            frame_save = save_path_pics + mvm_fname[:-8] + '_f'+str(fno)+'.png'
            utils.visualize_pose(videos_path+vid_fname, fno, newtracks, savepath=frame_save)
        with open(save_path + newtrack_fname, "wb") as outputfile:
            pickle.dump(newtracks, outputfile)
    mvm_vec_list = np.hstack(mvm_vec_list)

    whitevals = np.hstack(whitevals)
    blackvals = np.hstack(blackvals)
    plt.figure(figsize=(11,6))
    plt.title('Cross Sessions: White/Black ROI vals')
    plt.hist(whitevals, bins=50, color='blue', label='white mouse roi', alpha=0.7)
    plt.hist(blackvals, bins=50, color='red', label='black mouse roi', alpha=0.7)
    
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()