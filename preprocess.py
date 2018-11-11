import numpy as np
import pandas as pd
from scipy.signal import argrelmin, argrelmax, butter, lfilter, freqz
from scipy.fftpack import fft
pd.options.display.max_columns = None
pd.set_option('display.expand_frame_repr', False)

# Shared Functions
def rescale(input_list, rescale_size):
    skip = len(input_list) // rescale_size
    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    # Cut off the last one if needed.
    excess = len(output) - rescale_size

    return np.array(output[int(excess / 2):int(excess / 2) + rescale_size]) # Take middle values


def smoothen(data, smoothen_span):
    ewma = pd.Series.ewm
    fwd = ewma(pd.Series(data), span=smoothen_span).mean()
    bwd = ewma(pd.Series(data)[::-1], span=smoothen_span).mean()

    return np.array((fwd + bwd[::-1]) / 2)


def smooth_data(data, smoothen_span, data_var):
    for i in range(0, len(data['index1'].unique())):
        temp_df = data[data['index1'] == data['index1'].unique()[i]].copy()

        for var in data_var:
            temp_df[var] = smoothen(temp_df[var], smoothen_span)

        if i == 0:
            data_smooth = temp_df.copy()
        else:
            data_smooth = data_smooth.append(temp_df, ignore_index=True)

    return data_smooth


# Inertial Functions
def preprocess_inertial(inertial, rescale_size, smoothen_span):
    for i in range(0,len(inertial['index1'].unique())):
        print(i)
        x_accel = smoothen(rescale(inertial['x-accel'][inertial['index1']==inertial['index1'].unique()[i]].values, rescale_size), smoothen_span)
        y_accel = smoothen(rescale(inertial['y-accel'][inertial['index1'] == inertial['index1'].unique()[i]].values, rescale_size), smoothen_span)
        z_accel = smoothen(rescale(inertial['z-accel'][inertial['index1'] == inertial['index1'].unique()[i]].values, rescale_size), smoothen_span)
        x_gyro = smoothen(rescale(inertial['x-gyro'][inertial['index1'] == inertial['index1'].unique()[i]].values, rescale_size), smoothen_span)
        y_gyro = smoothen(rescale(inertial['y-gyro'][inertial['index1'] == inertial['index1'].unique()[i]].values, rescale_size), smoothen_span)
        z_gyro = smoothen(rescale(inertial['z-gyro'][inertial['index1'] == inertial['index1'].unique()[i]].values, rescale_size), smoothen_span)

        temp_df = pd.DataFrame({'x_acc':x_accel,
                                'y_acc':y_accel,
                                'z_acc':z_accel,
                                'x_gyro':x_gyro,
                                'y_gyro':y_gyro,
                                'z_gyro':z_gyro})

        temp_df['action'] = inertial['index1'].unique()[i][0]
        temp_df['subject'] = inertial['index1'].unique()[i][1]
        temp_df['trial'] = inertial['index1'].unique()[i][2]
        temp_df['inertial_frame'] = np.arange(1, rescale_size+1)

        if i == 0:
            inertial2 = temp_df.copy()
        else:
            inertial2 = inertial2.append(temp_df, ignore_index=True)

    return inertial2


def butter_lowpass(cutoff, fs, order):
    # helper function to return coefficients for scipy.lfilter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def get_gravity_component(data, cutoff, fs, order=5):
    # apply lowpass filter
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order):
    # helper function to return coefficients for scipy.lfilter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def get_body_component(data, cutoff, fs, order=5):
    # apply highpass filter
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def create_inertial_features(data):
    for i in range(0, len(data['index1'].unique())):
        print(i)

        temp_df = data[data['index1'] == data['index1'].unique()[i]].copy()

        # Seperate acceleration data using high and low pass
        #temp_df['x_acc_body'] = get_body_component(temp_df['x_acc'], 0.2, 50, 3)
        #temp_df['y_acc_body'] = get_body_component(temp_df['y_acc'], 0.2, 50, 3)
        #temp_df['z_acc_body'] = get_body_component(temp_df['z_acc'], 0.2, 50, 3)
        #temp_df['x_acc_gravity'] = get_gravity_component(temp_df['x_acc'], 0.2, 50, 3)
        #temp_df['y_acc_gravity'] = get_gravity_component(temp_df['y_acc'], 0.2, 50, 3)
        #temp_df['z_acc_gravity'] = get_gravity_component(temp_df['z_acc'], 0.2, 50, 3)

        # Calculate moving differences
        temp_df['x_acc_diff'] = temp_df['x_acc'].diff().fillna(0).values
        temp_df['y_acc_diff'] = temp_df['y_acc'].diff().fillna(0).values
        temp_df['z_acc_diff'] = temp_df['z_acc'].diff().fillna(0).values
        temp_df['x_gyro_diff'] = temp_df['x_acc'].diff().fillna(0).values
        temp_df['y_gyro_diff'] = temp_df['y_acc'].diff().fillna(0).values
        temp_df['z_gyro_diff'] = temp_df['z_acc'].diff().fillna(0).values
        #temp_df['x_acc_body_diff'] = temp_df['x_acc_body'].diff().fillna(0).values
        #temp_df['y_acc_body_diff'] = temp_df['y_acc_body'].diff().fillna(0).values
        #temp_df['z_acc_body_diff'] = temp_df['z_acc_body'].diff().fillna(0).values
        #temp_df['x_acc_gravity_diff'] = temp_df['x_acc_gravity'].diff().fillna(0).values
        #temp_df['y_acc_gravity_diff'] = temp_df['y_acc_gravity'].diff().fillna(0).values
        #temp_df['z_acc_gravity_diff'] = temp_df['z_acc_gravity'].diff().fillna(0).values

        # Calculate magnitude features
        temp_df['acc_mag'] = (temp_df['x_acc'] ** 2 + temp_df['y_acc'] ** 2 + temp_df['z_acc'] ** 2) ** 0.5
        temp_df['gyro_mag'] = (temp_df['x_gyro'] ** 2 + temp_df['y_gyro'] ** 2 + temp_df['z_gyro'] ** 2) ** 0.5
        temp_df['acc_diff_mag'] = (temp_df['x_acc_diff'] ** 2 + temp_df['y_acc_diff'] ** 2 + temp_df['z_acc_diff'] ** 2) ** 0.5
        temp_df['gyro_diff_mag'] = (temp_df['x_gyro_diff'] ** 2 + temp_df['y_gyro_diff'] ** 2 + temp_df['z_gyro_diff'] ** 2) ** 0.5
        #temp_df['acc_body_mag'] = (temp_df['x_acc_body'] ** 2 + temp_df['y_acc_body'] ** 2 + temp_df['z_acc_body'] ** 2) ** 0.5
        #temp_df['acc_gravity_mag'] = (temp_df['x_acc_gravity'] ** 2 + temp_df['y_acc_gravity'] ** 2 + temp_df['z_acc_gravity'] ** 2) ** 0.5

        if i == 0:
            data_engineered = temp_df.copy()
        else:
            data_engineered = data_engineered.append(temp_df, ignore_index=True)

    return data_engineered


def create_fft_features(data):
    for i in range(0, len(data['index1'].unique())):
        print(i)

        temp_df = data[data['index1'] == data['index1'].unique()[i]].copy()
        # filter = temp_df['index1'] == data['index1'].unique()[i]

        N = temp_df.shape[0]
        x_acc_fft = 2.0 / N * np.abs(fft(temp_df['x_acc'])[:N // 2])
        y_acc_fft = 2.0 / N * np.abs(fft(temp_df['y_acc'])[:N // 2])
        z_acc_fft = 2.0 / N * np.abs(fft(temp_df['z_acc'])[:N // 2])
        acc_mag_fft = 2.0 / N * np.abs(
            fft((temp_df['x_acc'] ** 2 + temp_df['y_acc'] ** 2 + temp_df['z_acc'] ** 2) ** 0.5)[:N // 2])

        x_gyro_fft = 2.0 / N * np.abs(fft(temp_df['x_gyro'])[:N // 2])
        y_gyro_fft = 2.0 / N * np.abs(fft(temp_df['y_gyro'])[:N // 2])
        z_gyro_fft = 2.0 / N * np.abs(fft(temp_df['z_gyro'])[:N // 2])
        gyro_mag_fft = 2.0 / N * np.abs(
            fft((temp_df['x_gyro'] ** 2 + temp_df['y_gyro'] ** 2 + temp_df['z_gyro'] ** 2) ** 0.5)[:N // 2])

        # Normalizing of fft variables
        x_acc_fft = x_acc_fft / x_acc_fft.sum()
        y_acc_fft = y_acc_fft / y_acc_fft.sum()
        z_acc_fft = z_acc_fft / z_acc_fft.sum()
        acc_mag_fft = acc_mag_fft / acc_mag_fft.sum()

        x_gyro_fft = x_gyro_fft / x_gyro_fft.sum()
        y_gyro_fft = y_gyro_fft / y_gyro_fft.sum()
        z_gyro_fft = z_gyro_fft / z_gyro_fft.sum()
        gyro_mag_fft = gyro_mag_fft / gyro_mag_fft.sum()

        # for var in [x_acc_fft, y_acc_fft, z_acc_fft,
        #             x_gyro_fft, y_gyro_fft, z_gyro_fft,
        #             acc_mag_fft, gyro_mag_fft]:
        #     for j in range(0,50,2):
        #         fft_array = np.array([var[j:j+2].min(),
        #                                    var[j:j+2].mean(),
        #                                    var[j:j+2].max(),
        #                                    var[j:j+2].std()])
        #         if j == 0:
        #             fft_feat = fft_array
        #         else:
        #             fft_feat = np.concatenate((fft_feat, fft_array))

        fft_feat = np.concatenate((x_acc_fft, y_acc_fft, z_acc_fft,
                                   x_gyro_fft, y_gyro_fft, z_gyro_fft,
                                   acc_mag_fft, gyro_mag_fft))

        temp_fft = pd.DataFrame(fft_feat).T
        temp_fft['action'] = temp_df['action'].values[0]
        temp_fft['subject'] = temp_df['subject'].values[0]
        temp_fft['trial'] = temp_df['trial'].values[0]

        if i == 0:
            data_fft = temp_fft.copy()
        else:
            data_fft = data_fft.append(temp_fft, ignore_index=True)

    return data_fft[data_fft.columns.tolist()[-3:] + data_fft.columns.tolist()[:-3]]


# Skeleton Functions
def preprocess_skeleton(skeleton, rescale_size, smoothen_span):
    for i in range(0, len(skeleton['index1'].unique())):
        print(i)

        skeleton_var = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12',
                        'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'y1', 'y2', 'y3',
                        'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10', 'y11', 'y12', 'y13', 'y14', 'y15',
                        'y16', 'y17', 'y18', 'y19', 'y20', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7',
                        'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15', 'z16', 'z17', 'z18', 'z19', 'z20']

        temp1_df = skeleton[skeleton['index1'] == skeleton['index1'].unique()[i]]

        table = []
        for var in skeleton_var:
            table.append(smoothen(rescale(temp1_df[var].values, rescale_size),smoothen_span))

        temp_df = pd.DataFrame(np.array(table).T, columns=skeleton_var)

        temp_df['action'] = skeleton['index1'].unique()[i][0]
        temp_df['subject'] = skeleton['index1'].unique()[i][1]
        temp_df['trial'] = skeleton['index1'].unique()[i][2]
        temp_df['skeleton_frame'] = np.arange(1, rescale_size+1)

        if i == 0:
            skeleton2 = temp_df.copy()
        else:
            skeleton2 = skeleton2.append(temp_df, ignore_index=True)

    return skeleton2


def euclidean(data, part1, part2):
    return (((data['x'+part1]-data['x'+part2])**2 +\
            (data['y'+part1]-data['y'+part2])**2 +\
            (data['z'+part1]-data['z'+part2])**2)**0.5)


def angle_between(u, v):

    ang1 = np.arctan2(*u[::-1])
    ang2 = np.arctan2(*v[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def xy_angle_between(data, part1, part2, part3):
    ax, bx, cx, ay, by, cy = data['x' + part1].values, data['x' + part2].values, data['x' + part3].values, \
                             data['y' + part1].values, data['y' + part2].values, data['y' + part3].values

    angles = []
    for i in range(len(ax)):
        A = np.array([ax[i], ay[i]])
        B = np.array([bx[i], by[i]])
        C = np.array([cx[i], cy[i]])

        # Convert the points to radians space
        a = np.radians(np.array(A))
        b = np.radians(np.array(B))
        c = np.radians(np.array(C))

        # Vectors in radians space
        avec = b - a
        cvec = c - b

        angle = angle_between(avec, cvec) * np.pi / 360
        angles.append(angle)

    return np.array(angles)


def yz_angle_between(data, part1, part2, part3):
    ay, by, cy, az, bz, cz = data['y' + part1].values, data['y' + part2].values, data['y' + part3].values, \
                             data['z' + part1].values, data['z' + part2].values, data['z' + part3].values

    angles = []
    for i in range(len(ay)):
        A = np.array([ay[i], az[i]])
        B = np.array([by[i], bz[i]])
        C = np.array([cy[i], cz[i]])

        # Convert the points to numpy latitude/longitude radians space
        a = np.radians(np.array(A))
        b = np.radians(np.array(B))
        c = np.radians(np.array(C))

        # Convert the points to numpy latitude/longitude radians space
        a = np.radians(np.array(A))
        b = np.radians(np.array(B))
        c = np.radians(np.array(C))

        # Vectors in latitude/longitude space
        avec = b - a
        cvec = c - b

        angle = angle_between(avec, cvec) * np.pi / 360
        angles.append(angle)

    return np.array(angles)


def create_skeleton_features(data):
    for i in range(0, len(data['index1'].unique())):
        print(i)

        temp_df = data[data['index1'] == data['index1'].unique()[i]].copy()
        # filter = temp_df['index1'] == data['index1'].unique()[i]

        # Get change in key joint positions
        temp_df['lhand_x_diff'] = temp_df['x8'].values - temp_df['x8'].values[0]
        temp_df['lhand_y_diff'] = temp_df['y8'].values - temp_df['y8'].values[0]
        temp_df['lhand_z_diff'] = temp_df['z8'].values - temp_df['z8'].values[0]
        temp_df['rhand_x_diff'] = temp_df['x12'].values - temp_df['x12'].values[0]
        temp_df['rhand_y_diff'] = temp_df['y12'].values - temp_df['y12'].values[0]
        temp_df['rhand_z_diff'] = temp_df['z12'].values - temp_df['z12'].values[0]
        temp_df['lfoot_x_diff'] = temp_df['x16'].values - temp_df['x16'].values[0]
        temp_df['lfoot_y_diff'] = temp_df['y16'].values - temp_df['y16'].values[0]
        temp_df['lfoot_z_diff'] = temp_df['z16'].values - temp_df['z16'].values[0]
        temp_df['rfoot_x_diff'] = temp_df['x20'].values - temp_df['x20'].values[0]
        temp_df['rfoot_y_diff'] = temp_df['y20'].values - temp_df['y20'].values[0]
        temp_df['rfoot_z_diff'] = temp_df['z20'].values - temp_df['z20'].values[0]
        temp_df['hip_x_diff'] = temp_df['x4'].values - temp_df['x4'].values[0]
        temp_df['hip_y_diff'] = temp_df['y4'].values - temp_df['y4'].values[0]
        temp_df['hip_z_diff'] = temp_df['z4'].values - temp_df['z4'].values[0]
        temp_df['head_x_diff'] = temp_df['x1'].values - temp_df['x1'].values[0]
        temp_df['head_y_diff'] = temp_df['y1'].values - temp_df['y1'].values[0]
        temp_df['head_z_diff'] = temp_df['z1'].values - temp_df['z1'].values[0]

        # Get distance between key joint positions
        temp_df['dist_hands'] = euclidean(temp_df, '8', '12') / euclidean(temp_df, '1', '4').median()
        temp_df['dist_feet'] = euclidean(temp_df, '16', '20') / euclidean(temp_df, '1', '4').median()
        temp_df['dist_lhand_lfoot'] = euclidean(temp_df, '8', '16') / euclidean(temp_df, '1', '4').median()
        temp_df['dist_rhand_rfoot'] = euclidean(temp_df, '12', '20') / euclidean(temp_df, '1', '4').median()
        temp_df['dist_lhand_head'] = euclidean(temp_df, '8', '1') / euclidean(temp_df, '1', '4').median()
        temp_df['dist_rhand_head'] = euclidean(temp_df, '12', '1') / euclidean(temp_df, '1', '4').median()
        temp_df['dist_rhand_lfoot'] = euclidean(temp_df, '12', '16') / euclidean(temp_df, '1', '4').median()
        temp_df['dist_lhand_rfoot'] = euclidean(temp_df, '8', '20') / euclidean(temp_df, '1', '4').median()
        temp_df['dist_head_lfoot'] = euclidean(temp_df, '1', '16') / euclidean(temp_df, '1', '4').median()
        temp_df['dist_head_rfoot'] = euclidean(temp_df, '1', '20') / euclidean(temp_df, '1', '4').median()

        # Get angle between joint triplets
        temp_df['xy_angle_lankle_lknee_lhip'] = xy_angle_between(temp_df, '15', '14', '13')
        temp_df['xy_angle_rankle_rknee_rhip'] = xy_angle_between(temp_df, '17', '18', '19')
        temp_df['xy_angle_rwrist_relbow_rshoulder'] = xy_angle_between(temp_df, '11', '10', '9')
        temp_df['xy_angle_lwrist_lelbow_lshoulder'] = xy_angle_between(temp_df, '7', '6', '5')
        temp_df['xy_angle_rshoulder_rknee_rhip'] = xy_angle_between(temp_df, '9', '18', '17')
        temp_df['xy_angle_lshoulder_lknee_lhip'] = xy_angle_between(temp_df, '5', '13', '12')

        temp_df['xy_angle_lfoot_lknee_hip'] = xy_angle_between(temp_df, '16', '14', '4')
        temp_df['xy_angle_rfoot_rknee_hip'] = xy_angle_between(temp_df, '20', '18', '4')
        temp_df['xy_angle_rhand_relbow_rshoulder'] = xy_angle_between(temp_df, '12', '10', '9')
        temp_df['xy_angle_lhand_lelbow_lshoulder'] = xy_angle_between(temp_df, '8', '6', '5')
        temp_df['xy_angle_lelbow_hip_lknee'] = xy_angle_between(temp_df, '6', '4', '14')
        temp_df['xy_angle_relbow_hip_rknee'] = xy_angle_between(temp_df, '10', '4', '18')

        temp_df['yz_angle_lankle_lknee_lhip'] = yz_angle_between(temp_df, '15', '14', '13')
        temp_df['yz_angle_rankle_rknee_rhip'] = yz_angle_between(temp_df, '17', '18', '19')
        temp_df['yz_angle_rwrist_relbow_rshoulder'] = yz_angle_between(temp_df, '11', '10', '9')
        temp_df['yz_angle_lwrist_lelbow_lshoulder'] = yz_angle_between(temp_df, '7', '6', '5')
        temp_df['yz_angle_rshoulder_rknee_rhip'] = yz_angle_between(temp_df, '9', '18', '17')
        temp_df['yz_angle_lshoulder_lknee_lhip'] = yz_angle_between(temp_df, '5', '13', '12')

        temp_df['yz_angle_lfoot_lknee_hip'] = yz_angle_between(temp_df, '16', '14', '4')
        temp_df['yz_angle_rfoot_rknee_hip'] = yz_angle_between(temp_df, '20', '18', '4')
        temp_df['yz_angle_rhand_relbow_rshoulder'] = yz_angle_between(temp_df, '12', '10', '9')
        temp_df['yz_angle_lhand_lelbow_lshoulder'] = yz_angle_between(temp_df, '8', '6', '5')
        temp_df['yz_angle_lelbow_hip_lknee'] = yz_angle_between(temp_df, '6', '4', '14')
        temp_df['yz_angle_relbow_hip_rknee'] = yz_angle_between(temp_df, '10', '4', '18')

        if i == 0:
            data_engineered = temp_df.copy()
        else:
            data_engineered = data_engineered.append(temp_df, ignore_index=True)

    return data_engineered


### Preparing Inertial Data
inertial = pd.read_csv('inertial_parsed.csv', header=0)
inertial['index1'] = list(zip(inertial['action'], inertial['subject'], inertial['trial']))
inertial_ref = inertial[['action', 'subject', 'trial', 'index1', 'inertial_frame']].groupby(['action', 'subject', 'trial', 'index1']).aggregate('count').reset_index()
inertial_ref['inertial_frame'].describe()

inertial = preprocess_inertial(inertial, 107, 10)

# Creating six temporal frames within each trial
inertial['inertial_bins'] = inertial.groupby(['action', 'subject', 'trial'])[['inertial_frame']].transform(lambda x: pd.cut(x, bins=6, labels=['B1','B2','B3','B4','B5','B6']))
inertial['index1'] = list(zip(inertial['action'], inertial['subject'], inertial['trial']))


# Inertial Feature Engineering
inertial_fft = create_fft_features(inertial)
# inertial_fft.to_csv('inertial_fft_clean.csv', index=False)

inertial = create_inertial_features(inertial)

main_var = ['action', 'subject', 'trial', 'inertial_bins',
            'x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro',
            'x_acc_diff', 'y_acc_diff', 'z_acc_diff',
            'x_gyro_diff', 'y_gyro_diff', 'z_gyro_diff',
            'acc_mag', 'gyro_mag', 'acc_diff_mag', 'gyro_diff_mag']
group_var = ['action', 'subject', 'trial', 'inertial_bins']
inertial2a = inertial[main_var].groupby(group_var).aggregate(['mean', 'std', 'min', 'max'])
inertial2a.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in inertial2a.columns]
inertial2b = inertial.copy()
inertial_var = ['x_acc', 'y_acc', 'z_acc', 'x_gyro', 'y_gyro', 'z_gyro',
                'x_acc_diff', 'y_acc_diff', 'z_acc_diff',
                'x_gyro_diff', 'y_gyro_diff', 'z_gyro_diff',
                'acc_mag', 'gyro_mag', 'acc_diff_mag', 'gyro_diff_mag']
inertial2b[inertial_var] = inertial2b[inertial_var]**2
inertial2b = inertial2b[main_var].groupby(group_var).mean()**0.5
inertial2b.columns = [col + '_rms' for col in inertial2b.columns]
inertial2 = pd.concat([inertial2a,inertial2b], axis=1)

inertial3 = pd.pivot_table(inertial2.reset_index(),
                           index=['action','subject','trial'],
                           columns='inertial_bins').reset_index()
inertial3.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in inertial3.columns]

# inertial3.to_csv('inertial_clean.csv', index=False)
# inertial3 = pd.read_csv('inertial_clean.csv', header=0)


### Preparing Skeleton Data
skeleton = pd.read_csv('skeleton_parsed.csv', header=0)
skeleton = pd.pivot_table(skeleton,
                          index=['action', 'subject', 'trial', 'skeleton_frame'],
                          columns='skeleton_joint').reset_index()

skeleton.columns =[s1 + str(s2) for (s1,s2) in skeleton.columns.tolist()]

skeleton.columns = skeleton.columns.str.replace('\.0', '', regex=True)
skeleton['index1'] = list(zip(skeleton['action'], skeleton['subject'], skeleton['trial']))
skeleton_ref = skeleton[['action', 'subject', 'trial', 'index1', 'skeleton_frame']].groupby(['action', 'subject', 'trial', 'index1']).aggregate('count').reset_index()

skeleton_ref['skeleton_frame'].describe()

skeleton = preprocess_skeleton(skeleton, 41, 4)
skeleton['index1'] = list(zip(skeleton['action'], skeleton['subject'], skeleton['trial']))

# Creating six temporal frames within each trial
skeleton['skeleton_bins'] = skeleton.groupby(['action', 'subject', 'trial'])[['skeleton_frame']].transform(lambda x: pd.cut(x, bins=6, labels=['B1','B2','B3','B4','B5','B6']))

# Skeleton Feature Engineering
skeleton = create_skeleton_features(skeleton)

main_var = ['action', 'subject', 'trial', 'skeleton_bins',
            'x8', 'x12', 'x14', 'x16', 'x18', 'x20', 'x4', 'x1',
            'y8', 'y12', 'y14', 'y16', 'y18', 'y20', 'y4', 'y1',
            'z8', 'z12', 'z14', 'z16', 'z20', 'z20', 'z4', 'z1',
            'xy_angle_lankle_lknee_lhip', 'xy_angle_rankle_rknee_rhip', 'xy_angle_rwrist_relbow_rshoulder',
            'xy_angle_lwrist_lelbow_lshoulder', 'xy_angle_rshoulder_rknee_rhip', 'xy_angle_lshoulder_lknee_lhip',
            'yz_angle_lankle_lknee_lhip', 'yz_angle_rankle_rknee_rhip', 'yz_angle_rwrist_relbow_rshoulder',
            'yz_angle_lwrist_lelbow_lshoulder', 'yz_angle_rshoulder_rknee_rhip', 'yz_angle_lshoulder_lknee_lhip',
            'lhand_x_diff', 'lhand_y_diff', 'lhand_z_diff',
            'rhand_x_diff', 'rhand_y_diff', 'rhand_z_diff',
            'lfoot_x_diff', 'lfoot_y_diff', 'lfoot_z_diff',
            'rfoot_x_diff', 'rfoot_y_diff', 'rfoot_z_diff',
            'hip_x_diff', 'hip_y_diff', 'hip_z_diff',
            'head_x_diff', 'head_y_diff', 'head_z_diff',
            'dist_hands', 'dist_feet', 'dist_lhand_lfoot', 'dist_rhand_rfoot',
            'dist_lhand_head', 'dist_rhand_head', 'dist_rhand_lfoot',
            'dist_lhand_rfoot', 'dist_head_lfoot', 'dist_head_rfoot',
            'xy_angle_lfoot_lknee_hip', 'xy_angle_rfoot_rknee_hip',
            'xy_angle_rhand_relbow_rshoulder', 'xy_angle_lhand_lelbow_lshoulder',
            'xy_angle_lelbow_hip_lknee', 'xy_angle_relbow_hip_rknee',
            'yz_angle_lfoot_lknee_hip', 'yz_angle_rfoot_rknee_hip',
            'yz_angle_rhand_relbow_rshoulder', 'yz_angle_lhand_lelbow_lshoulder',
            'yz_angle_lelbow_hip_lknee', 'yz_angle_relbow_hip_rknee']
group_var = ['action', 'subject', 'trial', 'skeleton_bins']
skeleton2a = skeleton[main_var].groupby(group_var).aggregate(['mean','std', 'min', 'max'])
skeleton2a.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in skeleton2a.columns]
skeleton3 = pd.pivot_table(skeleton2a.reset_index(),
                           index=['action', 'subject', 'trial'],
                           columns='skeleton_bins').reset_index()
skeleton3.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in skeleton3.columns]

# skeleton3.to_csv('skeleton_clean.csv', index=False)
# skeleton3 = pd.read_csv('skeleton_clean.csv', header=0)


### Creation of Inertial + Skeleton Dataset
inertial_skeleton = inertial3.merge(skeleton3,
                                    how='left',
                                    left_on=['action','subject','trial'],
                                    right_on=['action','subject','trial'])

# inertial_skeleton.to_csv('inertial_skeleton_clean.csv', index=False)
# inertial_skeleton = pd.read_csv('inertial_skeleton_clean.csv', header=0)