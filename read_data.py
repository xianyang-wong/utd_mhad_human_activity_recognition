import scipy.io
import os
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

def import_inertial_data(action, subject, trial):
    filename = os.path.join(os.getcwd(),'Inertial/a'+str(action)+'_s'+str(subject)+'_t'+str(trial)+'_inertial.mat')
    print(filename)
    if Path(filename).is_file():
        mat = scipy.io.loadmat(filename)
        return mat['d_iner']
    else:
        return None

def transform_inertial_data(action, subject, trial):
    data = import_inertial_data(action, subject, trial)
    if data is None: return None
    result = np.insert(data, 0, [[action], [subject], [trial]], axis=1)
    return np.array(result)

def transform_inertial_data_to_df(action, subject, trial):
    data = transform_inertial_data(action, subject, trial)
    if data is None: return None
    df = pd.DataFrame(data)
    df['record_no'] = range(1, df.shape[0] + 1, 1)
    df.columns = ['action', 'subject', 'trial', 'x-accel', 'y-accel', 'z-accel', 'x-gyro', 'y-gyro',
                  'z-gyro', 'inertial_frame']

    df = df[['action', 'subject', 'trial', 'inertial_frame', 'x-accel', 'y-accel', 'z-accel', 'x-gyro', 'y-gyro',
                  'z-gyro']]
    return df

def export_inertial_data_to_csv(action, subject, trial):
    df = transform_inertial_data_to_df(action, subject, trial)
    if df is None: return None
    filename = f'a{action}_s{subject}_t{trial}_inertial.csv'
    df.to_csv(filename, index=False)

actions_list = range(1,28,1)
subject_list = range(1,9,1)
trial_list = range(1,5,1)

for action in actions_list:
    for subject in subject_list:
        for trial in trial_list:
            record = transform_inertial_data_to_df(action, subject, trial)
            if (action==1) & (subject==1) & (trial==1):
                df = record
            else:
                df = df.append(record, ignore_index=True)

df.to_csv('inertial_parsed.csv',index=False)

def import_skeleton_data(action, subject, trial):
    filename = os.path.join(os.getcwd(), 'Skeleton/a' + str(action) + '_s' + str(subject) + '_t' + str(trial) + '_skeleton.mat')
    print(filename)
    if Path(filename).is_file():
        mat = scipy.io.loadmat(filename)
        return mat['d_skel']
    else:
        return None

def transform_skeleton_data(action, subject, trial):
    matrices = []
    data = import_skeleton_data(action, subject, trial)
    if data is None: return None
    for frame in range(data.shape[2]):
        skeleton_joints = [i + 1 for i in range(20)]
        matrix = data[:,:,frame]
        matrix = np.insert(matrix, 0, skeleton_joints, axis=1)
        matrix = np.insert(matrix, 0, frame, axis=1)
        matrices.append(matrix)
    result = np.vstack(tuple(matrices))
    result = np.insert(result, 0, [[action], [subject], [trial]], axis=1)
    return result

def transform_skeleton_data_to_df(action, subject, trial):
    data = transform_skeleton_data(action, subject, trial)
    if data is None: return None
    df = pd.DataFrame(data)
    df.columns = ['action', 'subject', 'trial', 'skeleton_frame', 'skeleton_joint', 'x', 'y', 'z']
    return df

actions_list = range(1,28,1)
subject_list = range(1,9,1)
trial_list = range(1,5,1)

for action in actions_list:
    for subject in subject_list:
        for trial in trial_list:
            record = transform_skeleton_data_to_df(action, subject, trial)
            if (action==1) & (subject==1) & (trial==1):
                df = record
            else:
                df = df.append(record, ignore_index=True)

df.to_csv('skeleton_parsed.csv',index=False)

def import_depth_data(action, subject, trial):
    filename = os.path.join(os.getcwd(), 'Depth/a' + str(action) + '_s' + str(subject) + '_t' + str(trial) + '_depth.mat')
    print(filename)
    if Path(filename).is_file():
        mat = scipy.io.loadmat(filename)
        return mat['d_depth']
    else:
        return None

def transform_depth_data(action, subject, trial):
    rows = []
    data = import_depth_data(action, subject, trial)
    if data is None: return None
    for frame in range(data.shape[2]):
        pixels = data[:, :, frame].flatten()
        rows.append(pixels)
    result = np.insert(rows, 0, [[action], [subject], [trial], [frame]], axis=1)
    return np.array(result)

def transform_depth_data_to_df(action, subject, trial):
    data = transform_depth_data(action, subject, trial)
    if data is None: return None
    df = pd.DataFrame(data)
    df.columns = ['action', 'subject', 'trial', 'depth_frame'] + [f'depth_{n}' for n in range(240 * 320)]
    return df

def export_depth_data_to_csv(action, subject, trial):
    df = transform_depth_data_to_df(action, subject, trial)
    if df is None: return None
    filename = f'a{action}_s{subject}_t{trial}_depth.csv'
    df.to_csv(filename, index=False)

def show_depth_image(action, subject, trial, frame):
    data = import_depth_data(action, subject, trial)
    if data is None: return None
    plt.imshow(data[:,:,frame], cmap='gray')
    plt.axis('off')
    plt.show()

actions_list = range(1,28,1)
subject_list = range(1,9,1)
trial_list = range(1,5,1)

for action in actions_list:
    for subject in subject_list:
        for trial in trial_list:
            print(str(action),str(subject),str(trial))
            record = transform_depth_data_to_df(action, subject, trial)
            if (action==1) & (subject==1) & (trial==1):
                df = record
            else:
                df = df.append(record, ignore_index=True)

df.to_csv('depth_parsed.csv',index=False)