
data_sets = {'Iris': 'datasets/Iris.csv', 'biodeg': 'datasets/biodeg.csv',
               'glass':'datasets/glass.csv', 'image segmentation':'datasets/image segmentation.csv',
               'Indian Liver Patient Dataset (ILPD)':'datasets/Indian Liver Patient Dataset (ILPD).csv', 'isolet':'datasets/isolet.csv',
               'magic04':'datasets/magic04.csv','movement_libras':'datasets/movement_libras.csv','wilt':'datasets/wilt.csv',
               'Wine_classification':'datasets/Wine_classification.csv'}

classifiers = ['RF', 'RFNB']

configurations = ['c1', 'c2', 'c3', 'c4']

configurations_paramters={
    'c1': {'n_estimators': 50, 'max_depth': 50, 'random_state': 0, 'min_samples_leaf': 1},
    'c2': {'n_estimators': 50, 'max_depth': 10, 'random_state': 0, 'min_samples_leaf': 5},
    'c3': {'n_estimators': 50, 'max_depth': 6, 'random_state': 0, 'min_samples_leaf': 15},
    'c4': {'n_estimators': 50, 'max_depth': 2, 'random_state': 0, 'min_samples_leaf': 25}}
