# imports
import h5py
#!apt-get update & apt-get install python-pydot python-pydot-ng graphviz -y
#!pip install livelossplot pydot graphviz tensorflow-addons ctgan pylab-sdk tqdm seaborn ipympl
from livelossplot import PlotLossesKerasTF
from livelossplot.outputs import MatplotlibPlot, ExtremaPrinter
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW, SGDW
import os, math, json, pylab, sys, time, imblearn
from imblearn.combine import SMOTEENN
from sklearn.metrics import mean_squared_error
plt.rcParams['figure.figsize'] = (24,4)
from sklearn.metrics import classification_report, confusion_matrix
from ctgan import CTGANSynthesizer
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import seaborn as sns
import sentry_sdk
sentry_sdk.init("https://ad563590bc1a4adfbb11a75c1548a3b2@o426400.ingest.sentry.io/5368180")


RESULTS_DIR    = './results'
CHECKPOINT_DIR = RESULTS_DIR + '/checkpoint'
REPORTS_DIR    = RESULTS_DIR + '/reports'
FIGS_DIR       = RESULTS_DIR + '/figs'

class CLSmodel():
    def __init__(self, DATASET, name, kfold=False):
        self.all_labels = ['BENIGN', 'DoS Hulk', 'SSH-Patator', 'PortScan', 'DoS GoldenEye', 
                  'DDoS', 'Heartbleed', 'Web Attack Brute Force', 'FTP-Patator', 
                  'Web Attack XSS', 'DoS slowloris', 'Infiltration', 'Bot', 
                  'Web Attack Sql Injection', 'DoS Slowhttptest']
        
        self.name                                            = name
        self.kfold                                           = kfold
        if not self.kfold:
            self.X_train, self.y_train, self.X_test, self.y_test = self.load_dataset(DATASET)
            self.initial_bias_calculate()
            self.model                                           = self.create_model()
            
        else:
            self.X, self.Y                                       = self.load_dataset_all(DATASET)
            self.numfolds                                        = 5
            self.reports                                         = {}
            self.cms                                             = {}
                

    def initial_bias_calculate(self):
        attacks                                              = self.X_train[self.y_train_l != 0].shape[0]
        total                                                = self.y_train_l.shape[0]
        self.initial_bias                                    = tf.keras.initializers.Constant(np.log([attacks / total]))

        self.total_numbers   = []
        for i in range(15):
            self.total_numbers.append(self.y_train_l[self.y_train_l == i].shape[0])

        self.total_numbers   = np.asarray(self.total_numbers)
        self.log_of_numbers  = np.log(self.total_numbers.min() / sum(self.total_numbers))
        self.initial_bias    = tf.keras.initializers.Constant(self.log_of_numbers)

    def confusionmatrix(self, y_pred):
        cm = confusion_matrix(self.y_test.argmax(axis=1), y_pred.argmax(axis=1))
        sns.heatmap(cm, annot=True, robust=True)
        plt.show()
            
                
    #### kfold cross validation automatic pipeline
    def pipeline(self, batch_size=10240, epochs=1, numfolds=None, upsampling='none'):
        self.numfolds = numfolds
        kfold         = StratifiedKFold(n_splits=self.numfolds, shuffle=True, random_state=42)
        
        self.current_kfold = 1
        
        start_time = time.time()
        for train_index, test_index in kfold.split(self.X,self.Y):
            sys.stdout.write("\r")
            print('\n###########\nfold #{} is in progress..\n###########\n'.format(self.current_kfold))
            
            # set train-test sets for this fold
            self.X_train_l = self.X[train_index].copy()
            self.X_test_l  = self.X[test_index].copy()
            
            self.y_train_l = self.Y[train_index].copy()
            self.y_test_l  = self.Y[test_index].copy()
            
            # in order to upsample heartbleed for 2-fold, it needs to have at least 6 samples for each fold. we need to copy this twice for each fold.
            if self.numfolds == 2 or upsampling == 'ctgan':
                self.X_train_l = np.concatenate([self.X_train_l, self.X[self.Y == 6]], axis=0)
                self.y_train_l = np.concatenate([self.y_train_l, self.Y[self.Y == 6]], axis=0)
                
                self.X_train_l = np.concatenate([self.X_train_l, self.X[self.Y == 13]], axis=0)
                self.y_train_l = np.concatenate([self.y_train_l, self.Y[self.Y == 13]], axis=0)
            #########
            
            self.X_train                  = self.X_train_l.reshape(self.X_train_l.shape[0], 1, self.X_train_l.shape[1])
            self.X_test                   = self.X_test_l.reshape(self.X_test_l.shape[0], 1, self.X_test_l.shape[1])
        
            self.y_train                  = tf.keras.utils.to_categorical(self.y_train_l)
            self.y_test                   = tf.keras.utils.to_categorical(self.y_test_l)
            
            # apply upsampling if applicable

            if upsampling == 'random' or upsampling == 'smoteenn' or upsampling == 'ctgan':
                
                # note that the number of samples of each class
                
                for xx in range(15):
                    print("len({}): {}".format(self.all_labels[xx], self.X_train[self.y_train_l == xx].shape[0]))
                
                # first find which classes should we apply and how much samples are going to needed
                
                ### order median ###
                """
                all_sets_tmp = {}
                for i in range(len(self.all_labels)):
                    all_sets_tmp[i]= self.X_train[self.y_train_l == i].shape[0]
                result = []
                for i in sorted(all_sets_tmp.items(), key=lambda x: x[1], reverse=True):
                    result.append((i[0], self.X_train[self.y_train_l == i[0]].shape[0]))

                desired_num_samples = result[int(len(result)/2)][1]
                
                apply = {}
                for i in result[int(len(result)/2)+1:]:
                    apply[i[0]] = desired_num_samples - i[1]       
                    
                """
                
                ### sample count median for attacks ###
                #"""
                all_sets_tmp = {}
                for i in range(len(self.all_labels)):
                    all_sets_tmp[i]= self.X_train[self.y_train_l == i].shape[0]
                result = 0
                for i in sorted(list(all_sets_tmp.items())[1:], key=lambda x: x[1], reverse=True):
                    result += self.X_train[self.y_train_l == i[0]].shape[0]

                desired_num_samples = int(result / len(list(all_sets_tmp.items())[1:]))
                
                
                apply = {}
                for i in list(all_sets_tmp.keys())[1:]:
                    samples = desired_num_samples - all_sets_tmp[i]
                    if samples > 0:
                        apply[i] = samples
                        
                #"""

                print("\n\n-----------\n")
                for xx in apply.keys():
                    print("{}: {}".format(self.all_labels[xx], apply[xx]))
                    
                print('\n\n-----------\n')

                self.add_synthetic(method=upsampling, apply=apply)
            else:
                print("\n\n-----------\n")
                print('No upsampling method found! Continuing for raw process...')
                print('\n\n-----------\n')
            
            # initial_bias calculation
            
            self.initial_bias_calculate()
            
            # create model as usual but specific for this particular fold 
            
            self.model = self.create_model()
            self.compile(batch_size=batch_size, epochs=epochs)
            self.fit()
            
            # load the best model found so far in this fold
            self.load_model('{}/classifier_{}_fold-{}.hdf5'.format(CHECKPOINT_DIR, self.name, self.current_kfold))
            
            # get the evaluation and save it to reports
            self.metrics()
            self.reports[self.current_kfold] = self.report
            self.cms[self.current_kfold]     = self.cm.tolist()
            
            self.current_kfold += 1
        
        # finish process!
        
        sys.stdout.write("\r")
        self.reports['time'] = time.time() - start_time
        
        print('\n###########\ndone in {} seconds\n###########\n'.format(self.reports['time']))
            
        # save reports to a file
        f1 = open('{}/classifier_{}_reports.json'.format(REPORTS_DIR, self.name),"w")
        f1.write(json.dumps(self.reports))
        f1.close()
        
        f2 = open('{}/classifier_{}_cms.json'.format(REPORTS_DIR, self.name),"w")
        f2.write(json.dumps(self.cms))
        f2.close()
        
        
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        
    def load_dataset_all(self, DATASET):
        ds = h5py.File(DATASET, 'r')
        self.X  = np.concatenate([ds['X_train'][:], ds['X_test'][:]], axis=0)
        self.Y  = np.concatenate([ds['y_train'][:], ds['y_test'][:]], axis=0)
            
        return (self.X, self.Y)
    
    def load_dataset(self, DATASET):
        ds = h5py.File(DATASET, 'r')
        self.X_train_l, self.X_test_l = ds['X_train'][:], ds['X_test'][:]
        self.y_train_l, self.y_test_l = ds['y_train'][:], ds['y_test'][:]
        
        self.X_train                  = self.X_train_l.reshape(self.X_train_l.shape[0], 1, self.X_train_l.shape[1])
        self.X_test                   = self.X_test_l.reshape(self.X_test_l.shape[0], 1, self.X_test_l.shape[1])
        
        
        self.y_train                  = tf.keras.utils.to_categorical(self.y_train_l)
        self.y_test                   = tf.keras.utils.to_categorical(self.y_test_l)
                
        return (self.X_train, self.y_train, self.X_test, self.y_test)
    
    
    # signature: add_synthetic(self=model_train, method='smoteenn', apply={9: 1000, 13: 890})
    def add_synthetic(self, method='random', apply={}):

        ################# random
        if method == 'random':
            sys.stdout.write("\r")
            print('\n###########\nadding random synthetic samples ..\n###########\n')
            print('\n{}\n'.format(apply))
            
            for i in tqdm(apply.keys()):
                noise_factor     = self.X_train[self.y_train_l == i].mean() * 0.001
                totalcount       = 0
                max_shape        = self.X_train[self.y_train_l == i].shape[0] + apply[i]
                for xx in range(3):
                    set_shape = self.X_train[self.y_train_l == i].shape[0]
                    if set_shape < max_shape:
                        howManyTimes = round(math.log(max_shape / set_shape)) + 1
                        for j in range(howManyTimes):
                            totalcount += 1
                            rareEventX   = self.X_train[self.y_train_l == i].copy()
                            rareEventY   = self.y_train_l[self.y_train_l == i].copy()

                            noisyRareEvent = rareEventX + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=rareEventX.shape)

                            if rareEventX.shape[0] + noisyRareEvent.shape[0] > max_shape:
                                will_be_subtracted = (rareEventX.shape[0] + noisyRareEvent.shape[0]) - max_shape
                                new_shape = noisyRareEvent.shape[0] - will_be_subtracted

                                self.X_train         = np.concatenate((self.X_train, noisyRareEvent[:new_shape]), axis=0)
                                self.y_train_l       = np.concatenate((self.y_train_l, rareEventY[:new_shape]), axis=0)
                            else:
                                self.X_train         = np.concatenate((self.X_train, noisyRareEvent), axis=0)
                                self.y_train_l       = np.concatenate((self.y_train_l, rareEventY), axis=0)
                                
                            self.y_train = tf.keras.utils.to_categorical(self.y_train_l)

                            print(self.all_labels[i], self.X_train[self.y_train_l == i].shape, " {}.th generation with {} noise".format(totalcount, noise_factor))
                            noise_factor *= 0.5
                        
        ################# smoteenn
        elif method == 'smoteenn':
            print('\n###########\nadding smoteenn synthetic samples ..\n###########\n')
            print('\n{}\n'.format(apply))
            
            competitors = apply.keys()
            sampling_strategy = {}
            for i in competitors:
                sampling_strategy[i] = int(self.y_train_l[self.y_train_l == i].shape[0] + apply[i]*1.5) 
            
            smote_enn = SMOTEENN(random_state=0, n_jobs=64, 
                                 sampling_strategy=sampling_strategy,
                                 enn=imblearn.under_sampling.EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=3)
                                )
            

            filters = [(self.y_train_l == i) for i in competitors]
                
            X_resampled, y_resampled = smote_enn.fit_resample(self.X_train_l[np.logical_or.reduce(filters)], 
                                                              self.y_train_l[np.logical_or.reduce(filters)])            
            
            # remove selected classes from dataset
            filters = [(self.y_train_l != i) for i in apply.keys()]
            
            previous_shapes = {}
            for i in apply.keys():
                previous_shapes[i] = self.y_train_l[self.y_train_l == i].shape[0]
            
            self.X_train_l = self.X_train_l[np.logical_and.reduce(filters)]
            self.y_train_l = self.y_train_l[np.logical_and.reduce(filters)]
            
            
            for i in apply.keys():
                self.X_train_l = np.concatenate([
                    self.X_train_l, 
                    X_resampled[y_resampled == i][:previous_shapes[i] + apply[i]]
                ], axis=0)
                self.y_train_l = np.concatenate([
                    self.y_train_l, 
                    y_resampled[y_resampled == i][:previous_shapes[i] + apply[i]]
                ], axis=0)

                
            self.X_train                  = self.X_train_l.reshape(self.X_train_l.shape[0], 1, self.X_train_l.shape[1])
            self.y_train                  = tf.keras.utils.to_categorical(self.y_train_l)
            
        elif method == 'ctgan':
            print('\n###########\nadding ctgan synthetic samples ..\n###########\n')
            print('\n{}\n'.format(apply))
            for i in tqdm(apply.keys()):
                gan_batch_size = 10
                if self.y_train_l[self.y_train_l == i].shape[0] > 100:
                    gan_batch_size = 20
                elif self.y_train_l[self.y_train_l == i].shape[0] > 300:
                    gan_batch_size = 50
                
                gan = CTGANSynthesizer(batch_size=gan_batch_size)
                gan.fit(self.X_train_l[self.y_train_l == i], epochs=100)
                
                # generate samples
                generated = gan.sample(apply[i])
                
                self.X_train_l = np.concatenate([self.X_train_l, generated], axis=0)
                self.X_train   = self.X_train_l.reshape(self.X_train_l.shape[0], 1, self.X_train_l.shape[1])
                
                self.y_train_l = np.concatenate([self.y_train_l, np.ones(shape=(apply[i],)) * i], axis=0)
                
                self.y_train = tf.keras.utils.to_categorical(self.y_train_l)
                
                
    
        
    def fully_connected(self, x):
        
        inputs   = x
        f_inputs = tf.keras.layers.Flatten(name='flatten1')(x)
        
        x  = tf.keras.layers.SeparableConv1D(filters=128, kernel_size=3, padding='same', activation='relu', 
                                             data_format='channels_last', name='convfeatures1_f')(inputs)
        x  = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
        
        x  = tf.keras.layers.SeparableConv1D(filters=256, kernel_size=3, padding='same', activation='relu', 
                                             data_format='channels_last', name='convfeatures2_f')(x)
        x  = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
        
        
        x  = tf.keras.layers.Flatten(name='flatten')(x)
        x  = tf.keras.layers.Dense(128, name='dense1', activation='selu', 
                                   kernel_initializer=tf.keras.initializers.lecun_normal())(x)
        x  = tf.keras.layers.BatchNormalization()(x)
        x  = tf.keras.layers.AlphaDropout(0.1)(x)
        
        
        x  = tf.keras.layers.Dense(64, name='dense2', activation='selu', 
                                   kernel_initializer=tf.keras.initializers.lecun_normal())(x)
        x  = tf.keras.layers.BatchNormalization()(x)
        x  = tf.keras.layers.AlphaDropout(0.1)(x)

        
        x  = tf.keras.layers.Dense(32, name='dense3', activation='selu', 
                                   kernel_initializer=tf.keras.initializers.lecun_normal())(x)   
        x  = tf.keras.layers.BatchNormalization()(x)
        x  = tf.keras.layers.AlphaDropout(0.1)(x)

        x  = tf.keras.layers.Concatenate()([f_inputs, x])
        
        x  = tf.keras.layers.Dense(15, activation=tf.nn.softmax, name='output', 
                                   kernel_initializer=tf.keras.initializers.lecun_normal(), bias_initializer=self.initial_bias)(x)   
        
        return x
    
    def create_model(self):
        timewindow  = self.X_train.shape[1]
        numfeatures = self.X_train.shape[2]

        inputs      = tf.keras.Input(shape=(timewindow, numfeatures))
        
        x           = self.fully_connected(inputs)
            
        self.model = tf.keras.Model(inputs=inputs, outputs=x, name=self.name)
        return self.model
    
    def summary(self):
        return self.model.summary()
    
    def compile(self, batch_size, epochs, loss='categorical_crossentropy', metrics=['accuracy']):
        self.batch_size = batch_size
        self.epochs     = epochs
        b, B, T         = batch_size, self.X_train.shape[0], epochs
        
        wd = 0.0025 * (b/B/T)**0.5
        self.model.compile(optimizer=AdamW(weight_decay=wd), loss=loss, metrics=['accuracy', tfa.metrics.F1Score(average='macro', num_classes=15)])
        print(self.summary())
        
    def fit(self, batch_size=None, epochs=None):
        try:
            if self.kfold:
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    '{}/classifier_{}_fold-{}.hdf5'.format(CHECKPOINT_DIR, self.name, self.current_kfold), 
                    monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')
            else:
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    '{}/classifier_{}.hdf5'.format(CHECKPOINT_DIR, self.name), 
                    monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')
            
            self.model.fit(self.X_train,
                           self.y_train,
                           epochs=epochs if epochs else self.epochs,
                           batch_size=batch_size if batch_size else self.batch_size,
                           validation_data=(self.X_test, self.y_test),
                           shuffle=True,
                           callbacks=[PlotLossesKerasTF(outputs=[MatplotlibPlot(cell_size=(6, 2), max_cols=3), ExtremaPrinter()]), checkpoint_callback])
        except KeyboardInterrupt:
            print('\n\n')
            print(self.model.history.history)
            
    
    def metrics(self):
        print("\n###########\npredicting...\n###########\n")
        y_pred = self.model.predict(self.X_test)
        
        self.report = classification_report(self.y_test_l, np.argmax(y_pred, 1), digits=6).split("\n")
        for i,r in enumerate(self.report):
            if i > 1 and i < 17:
                print (r, self.all_labels[int(r.split()[0])])
            else:
                print (r)
                
        counts, precisions, recalls, f1s = [], [], [], []
        for i in range(len(self.all_labels)):
            values = self.report[2:-3][i].split()
            counts.append(float(values[4]))
            precisions.append(float(values[1]))
            recalls.append(float(values[2]))
            f1s.append(float(values[3]))
        values = [counts, precisions, recalls, f1s]

        plt.figure()
        fig, axs = plt.subplots(1,1,figsize=(15,10))
        stats = ['sample count', 'precision', 'recall', 'f1',]
        for i, s in enumerate(stats):
            plt.subplot(len(stats)/4+1,4,i+1)
            plt.tight_layout()
            plt.plot(values[i], self.all_labels)
            plt.title(stats[i])
            
        plt.savefig('{}/classifier_{}_fold_{}_results.png'.format(FIGS_DIR, self.name, self.current_kfold), bbox_inches='tight')
        
        plt.show()
        
        plt.rcParams['figure.figsize'] = (24,10)
        
        plt.figure()
        self.cm = confusion_matrix(self.y_test.argmax(axis=1), y_pred.argmax(axis=1))
        
        sns.heatmap(self.cm, annot=True, robust=True)
        plt.savefig('{}/classifier_{}_fold_{}_cm.png'.format(FIGS_DIR, self.name, self.current_kfold), bbox_inches='tight')
        plt.show()