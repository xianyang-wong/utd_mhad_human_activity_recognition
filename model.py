import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pd.options.display.max_columns = None
pd.set_option('display.expand_frame_repr', False)


# Modeling Functions
def get_metrics(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)

    metrics = [model_name,
               accuracy_score(y_test, y_test_pred),
               precision_score(y_test, y_test_pred, average='weighted', labels=np.unique(y_test_pred)),
               recall_score(y_test, y_test_pred, average='weighted', labels=np.unique(y_test_pred)),
               f1_score(y_test, y_test_pred, average='weighted', labels=np.unique(y_test_pred))]

    return metrics


def feature_extract(X_train, y_train, n_feat):
    feature_classifier = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=612)
    feature_classifier.fit(X_train, y_train)

    feat_importances = pd.Series(feature_classifier.feature_importances_,
                                 index=X_train.columns)
    top_feat = feat_importances.nlargest(n_feat).reset_index()
    print(top_feat[0].sum())

    return top_feat['index'].tolist()


def generate_principal_components(data, n_components):
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(data.iloc[:, 3:])
    principalDF = pd.DataFrame(data=principalComponents,
                               columns=['pc' + str(i) for i in range(1, n_components + 1)])
    data_pc = pd.concat([data.iloc[:, :3], principalDF], axis=1)
    print(sum(pca.explained_variance_ratio_))

    return data_pc


def generate_modelling_outputs(data):
    X_train = data.iloc[:, 3:][data['subject'].isin([1, 3, 5, 7])]
    y_train = data['action'][data['subject'].isin([1, 3, 5, 7])]
    X_test = data.iloc[:, 3:][data['subject'].isin([2, 4, 6, 8])]
    y_test = data['action'][data['subject'].isin([2, 4, 6, 8])]

    X_train_f1 = data[data['subject'].isin([3, 5, 7])]
    y_train_f1 = data['action'][data['subject'].isin([3, 5, 7])]
    X_test_f1 = data[data['subject'].isin([1])]
    y_test_f1 = data['action'][data['subject'].isin([1])]

    X_train_f2 = data[data['subject'].isin([1, 5, 7])]
    y_train_f2 = data['action'][data['subject'].isin([1, 5, 7])]
    X_test_f2 = data[data['subject'].isin([3])]
    y_test_f2 = data['action'][data['subject'].isin([3])]

    X_train_f3 = data[data['subject'].isin([1, 3, 7])]
    y_train_f3 = data['action'][data['subject'].isin([1, 3, 7])]
    X_test_f3 = data[data['subject'].isin([5])]
    y_test_f3 = data['action'][data['subject'].isin([5])]

    X_train_f4 = data[data['subject'].isin([1, 3, 5])]
    y_train_f4 = data['action'][data['subject'].isin([1, 3, 5])]
    X_test_f4 = data[data['subject'].isin([7])]
    y_test_f4 = data['action'][data['subject'].isin([7])]

    lr_classifier = [LogisticRegression(multi_class='multinomial',
                                        C=0.01, solver='newton-cg', random_state=612), 'Logistic Regression']
    knn_classifier = [KNeighborsClassifier(n_neighbors=3), 'KNN']
    svm_classifier = [SVC(kernel='linear', C=0.01, random_state=612), 'SVM']
    rf_classifier = [RandomForestClassifier(n_estimators=500, min_samples_split=10, max_depth=10, max_features='sqrt', random_state=612),
                     'Random Forest']
    xt_classifier = [ExtraTreesClassifier(n_estimators=500, min_samples_split=10, max_depth=10, max_features='sqrt', random_state=612),
                     'Extra Trees']

    models = [lr_classifier, knn_classifier,
              svm_classifier, rf_classifier,
              xt_classifier]

    for model in range(0, len(models)):
        print("Generating Results for " + models[model][1])
        f1_metrics = get_metrics(models[model][0], models[model][1], X_train_f1, y_train_f1, X_test_f1, y_test_f1)
        f2_metrics = get_metrics(models[model][0], models[model][1], X_train_f2, y_train_f2, X_test_f2, y_test_f2)
        f3_metrics = get_metrics(models[model][0], models[model][1], X_train_f3, y_train_f3, X_test_f3, y_test_f3)
        f4_metrics = get_metrics(models[model][0], models[model][1], X_train_f4, y_train_f4, X_test_f4, y_test_f4)
        favg_metrics = [models[model][1],
                        np.min(np.array([f1_metrics[1], f2_metrics[1], f3_metrics[1], f4_metrics[1]])),
                        np.mean(np.array([f1_metrics[1], f2_metrics[1], f3_metrics[1], f4_metrics[1]])),
                        np.max(np.array([f1_metrics[1], f2_metrics[1], f3_metrics[1], f4_metrics[1]])),
                        np.std(np.array([f1_metrics[1], f2_metrics[1], f3_metrics[1], f4_metrics[1]])),
                        np.mean(np.array([f1_metrics[2], f2_metrics[2], f3_metrics[2], f4_metrics[2]])),
                        np.std(np.array([f1_metrics[2], f2_metrics[2], f3_metrics[2], f4_metrics[2]])),
                        np.mean(np.array([f1_metrics[3], f2_metrics[3], f3_metrics[3], f4_metrics[3]])),
                        np.std(np.array([f1_metrics[3], f2_metrics[3], f3_metrics[3], f4_metrics[3]])),
                        np.mean(np.array([f1_metrics[4], f2_metrics[4], f3_metrics[4], f4_metrics[4]])),
                        np.std(np.array([f1_metrics[4], f2_metrics[4], f3_metrics[4], f4_metrics[4]]))]
        test_metrics = get_metrics(models[model][0], models[model][1], X_train, y_train, X_test, y_test)

        row_metrics = favg_metrics + test_metrics[1:]
        if model == 0:
            output = pd.DataFrame(row_metrics).T
            output.columns = ['Model', 'CV-Accuracy-Min', 'CV-Accuracy-Mean', 'CV-Accuracy-Max',
                              'CV-Accuracy-StdDev', 'CV-Precision-Mean', 'CV-Precision-StdDev',
                              'CV-Recall-Mean', 'CV-Recall-StdDev',
                              'CV-F1 Measure-Mean', 'CV-F1 Measure-StdDev',
                              'Test-Accuracy', 'Test-Precision', 'Test-Recall', 'Test-F1 Measure']
        else:
            temp_output = pd.DataFrame(row_metrics).T
            temp_output.columns = ['Model', 'CV-Accuracy-Min', 'CV-Accuracy-Mean', 'CV-Accuracy-Max',
                                   'CV-Accuracy-StdDev', 'CV-Precision-Mean', 'CV-Precision-StdDev',
                                   'CV-Recall-Mean', 'CV-Recall-StdDev',
                                   'CV-F1 Measure-Mean', 'CV-F1 Measure-StdDev',
                                   'Test-Accuracy', 'Test-Precision', 'Test-Recall', 'Test-F1 Measure']
            output = output.append(temp_output, ignore_index=True)

    return output


def freq_stat(df):
    a = df.values
    zero_c = (a == 0).sum(1)
    one_c = a.shape[1] - zero_c
    df['max_vote'] = (zero_c <= one_c).astype(int)

    return df


def voting_ensemble(data):
    X_train = data.iloc[:, 3:][data['subject'].isin([1, 3, 5, 7])]
    y_train = data['action'][data['subject'].isin([1, 3, 5, 7])]
    X_test = data.iloc[:, 3:][data['subject'].isin([2, 4, 6, 8])]
    y_test = data['action'][data['subject'].isin([2, 4, 6, 8])]

    lr_classifier = LogisticRegression(multi_class='multinomial', C=0.01, solver='newton-cg', random_state=612)
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    svm_classifier = SVC(kernel='linear', C=0.01, random_state=612)
    rf_classifier = RandomForestClassifier(n_estimators=500, min_samples_split=10, max_depth=10, random_state=612)
    xt_classifier = ExtraTreesClassifier(n_estimators=500, min_samples_split=10, max_depth=10, max_features=10, random_state=612)

    lr_classifier.fit(X_train, y_train)
    y_test_pred_lr = lr_classifier.predict(X_test)

    knn_classifier.fit(X_train, y_train)
    y_test_pred_knn = knn_classifier.predict(X_test)

    svm_classifier.fit(X_train, y_train)
    y_test_pred_svm = svm_classifier.predict(X_test)

    rf_classifier.fit(X_train, y_train)
    y_test_pred_rf = rf_classifier.predict(X_test)

    xt_classifier.fit(X_train, y_train)
    y_test_pred_xt = xt_classifier.predict(X_test)

    pred_df = pd.DataFrame({'lr': y_test_pred_lr,
                            'knn': y_test_pred_knn,
                            'svm': y_test_pred_svm,
                            'rf': y_test_pred_rf,
                            'xt': y_test_pred_xt})

    pred_df['max_vote'] = pred_df.mode(axis=1)[0]

    ensemble_metrics = [accuracy_score(y_test, pred_df['max_vote']),
                        precision_score(y_test, pred_df['max_vote'], average='weighted', labels=np.unique(y_test)),
                        recall_score(y_test, pred_df['max_vote'], average='weighted', labels=np.unique(y_test)),
                        f1_score(y_test, pred_df['max_vote'], average='weighted', labels=np.unique(y_test))]

    return ensemble_metrics, y_test, pred_df['max_vote']


# Confusion matrix plotting function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #    plt.text(j, i, format(cm[i, j], fmt),
    #             horizontalalignment="center",
    #             color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Read in data
inertial = pd.read_csv('inertial_clean.csv', header=0)
inertial_fft = pd.read_csv('inertial_fft_clean.csv', header=0)
skeleton = pd.read_csv('skeleton_clean.csv', header=0)
inertial_skeleton = pd.read_csv('inertial_skeleton_clean.csv', header=0)

# Normalizing data
inertial.iloc[:,3:] = StandardScaler().fit_transform(inertial.iloc[:,3:])
skeleton.iloc[:,3:] = StandardScaler().fit_transform(skeleton.iloc[:,3:])
inertial_skeleton.iloc[:,3:] = StandardScaler().fit_transform(inertial_skeleton.iloc[:,3:])

# Inertial FFT Modelling
inertial_fft_top_feat = feature_extract(inertial_fft.iloc[:, 3:][inertial_fft['subject'].isin([1, 3, 5, 7])],
                                        inertial_fft['action'][inertial_fft['subject'].isin([1, 3, 5, 7])],200)
inertial_fft_output = generate_modelling_outputs(inertial_fft[['action','subject','trial']+inertial_fft_top_feat])
print(inertial_fft_output[['Model','CV-Accuracy-Mean','Test-Accuracy']].sort_values('Test-Accuracy',ascending=False))
inertial_fft_ensemble_metrics, inertial_fft_ensemble_y_test, inertial_fft_ensemble_max_vote = \
    voting_ensemble(inertial_fft[['action','subject','trial']+inertial_fft_top_feat])
print(inertial_fft_ensemble_metrics)

# Inertial + FFT
inertial_test = inertial.merge(inertial_fft,
                               how='left',
                               left_on=['action','subject','trial'],
                               right_on=['action','subject','trial'])

inertial_test_top_feat = feature_extract(inertial_test.iloc[:, 3:][inertial_test['subject'].isin([1, 3, 5, 7])],
                                    inertial_test['action'][inertial_test['subject'].isin([1, 3, 5, 7])],300)
inertial_test_output = generate_modelling_outputs(inertial_test[['action','subject','trial']+inertial_test_top_feat])
print(inertial_test_output[['Model','CV-Accuracy-Mean','Test-Accuracy']].sort_values('Test-Accuracy',ascending=False))
inertial_test_ensemble_metrics, inertial_test_ensemble_y_test, inertial_test_fft_ensemble_max_vote = \
    voting_ensemble(inertial_test[['action','subject','trial']+inertial_test_top_feat])
print(inertial_test_ensemble_metrics)

# Inertial Only
inertial_top_feat = feature_extract(inertial.iloc[:, 3:][inertial['subject'].isin([1, 3, 5, 7])],
                                    inertial['action'][inertial['subject'].isin([1, 3, 5, 7])],200)
inertial_output = generate_modelling_outputs(inertial[['action','subject','trial']+inertial_top_feat])
print(inertial_output[['Model','CV-Accuracy-Mean','Test-Accuracy']].sort_values('Test-Accuracy',ascending=False))
inertial_ensemble_metrics, inertial_ensemble_y_test, inertial_fft_ensemble_max_vote = \
    voting_ensemble(inertial[['action','subject','trial']+inertial_top_feat])
print(inertial_ensemble_metrics)

# Skeleton Feature Selection prior to training
skeleton_top_feat = feature_extract(skeleton.iloc[:, 3:][skeleton['subject'].isin([1, 3, 5, 7])],
                                    skeleton['action'][skeleton['subject'].isin([1, 3, 5, 7])],500)
skeleton_output = generate_modelling_outputs(skeleton[['action','subject','trial']+skeleton_top_feat])
print(skeleton_output[['Model','CV-Accuracy-Mean','Test-Accuracy']].sort_values('Test-Accuracy',ascending=False))
skeleton_ensemble_metrics, skeleton_ensemble_y_test, skeleton_ensemble_max_vote = \
    voting_ensemble(skeleton[['action','subject','trial']+skeleton_top_feat])
print(skeleton_ensemble_metrics)


# Inertial-Skeleton Feature Selection prior to training
inertial_skeleton_top_feat = feature_extract(inertial_skeleton.iloc[:, 3:][inertial_skeleton['subject'].isin([1, 3, 5, 7])],
                                             inertial_skeleton['action'][inertial_skeleton['subject'].isin([1, 3, 5, 7])], 500)
inertial_skeleton_output = generate_modelling_outputs(inertial_skeleton[['action','subject','trial']+inertial_skeleton_top_feat])
print(inertial_skeleton_output[['Model','CV-Accuracy-Mean','Test-Accuracy']].sort_values('Test-Accuracy',ascending=False))
inertial_skeleton_ensemble_metrics, inertial_skeleton_ensemble_y_test, inertial_skeleton_ensemble_max_vote = \
    voting_ensemble(inertial_skeleton[['action','subject','trial']+inertial_skeleton_top_feat])
print(inertial_skeleton_ensemble_metrics)


inertial_fft_output2 = inertial_fft_output[['Model','Test-Accuracy','Test-Precision','Test-Recall','Test-F1 Measure']].copy()
inertial_test_output2 = inertial_test_output[['Model','Test-Accuracy','Test-Precision','Test-Recall','Test-F1 Measure']].copy()
inertial_output2 = inertial_output[['Model','Test-Accuracy','Test-Precision','Test-Recall','Test-F1 Measure']].copy()
skeleton_output2 = skeleton_output[['Model','Test-Accuracy','Test-Precision','Test-Recall','Test-F1 Measure']].copy()
inertial_skeleton_output2 = inertial_skeleton_output[['Model','Test-Accuracy','Test-Precision','Test-Recall','Test-F1 Measure']].copy()

output2 = inertial_fft_output2.append(inertial_test_output2, ignore_index=True).append(inertial_output2, ignore_index=True)\
    .append(skeleton_output2, ignore_index=True).append(inertial_skeleton_output2, ignore_index=True)


### Plotting Functions
cm = confusion_matrix(inertial_skeleton_ensemble_y_test,
                      inertial_skeleton_ensemble_max_vote)

plot_confusion_matrix(cm,
                      classes=['Swipe left','Swipe right','Wave','Clap',
                               'Throw','Arm cross','Basketball shoot','Draw X',
                               'Draw circle (clockwise)','Draw circle (counter clockwise)',
                               'Draw triangle','Bowling','Boxing','Baseball swing',
                               'Tennis swing','Arm curl','Tennis serve','Push',
                               'Knock','Catch','Pickup and throw','Jog','Walk',
                               'Sit to stand','Stand to sit','Lunge','Squat'], normalize=True)

