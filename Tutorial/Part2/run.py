import os
import shutil
import pickle
import warnings
import pandas as pd
import numpy as np

import torch
from torch import optim as optimizers
from torch.nn import Sequential, Linear, Dropout, Flatten, LSTM

########################################################
#                   NAS PARAMETERS                     #
########################################################
CONTROLLER_SAMPLING_EPOCHS = 10
SAMPLES_PER_CONTROLLER_EPOCH = 10
CONTROLLER_TRAINING_EPOCHS = 10
ARCHITECTURE_TRAINING_EPOCHS = 10
CONTROLLER_LOSS_ALPHA = 0.9

########################################################
#               CONTROLLER PARAMETERS                  #
########################################################
CONTROLLER_LSTM_DIM = 100
CONTROLLER_OPTIMIZER = 'Adam'
CONTROLLER_LEARNING_RATE = 0.01
CONTROLLER_DECAY = 0.1
CONTROLLER_MOMENTUM = 0.0
CONTROLLER_USE_PREDICTOR = True

########################################################
#                   MLP PARAMETERS                     #
########################################################
MAX_ARCHITECTURE_LENGTH = 3
MLP_OPTIMIZER = 'Adam'
MLP_LEARNING_RATE = 0.01
MLP_DECAY = 0.0
MLP_MOMENTUM = 0.0
MLP_DROPOUT = 0.2
MLP_LOSS_FUNCTION = 'categorical_crossentropy'
MLP_ONE_SHOT = True

########################################################
#                   DATA PARAMETERS                    #
########################################################
TARGET_CLASSES = 3

########################################################
#                  OUTPUT PARAMETERS                   #
########################################################
TOP_N = 5

class MLPSearchSpace(object):

    def __init__(self, target_classes):

        self.target_classes = target_classes
        self.vocab = self.vocab_dict()

    def vocab_dict(self):
        nodes = [8, 16, 32, 64, 128, 256, 512]
        act_funcs = ['sigmoid', 'tanh', 'relu', 'elu']
        layer_params = []
        layer_id = []
        for i in range(len(nodes)):
            for j in range(len(act_funcs)):
                layer_params.append((nodes[i], act_funcs[j]))
                layer_id.append(len(act_funcs) * i + j + 1)
        vocab = dict(zip(layer_id, layer_params))
        vocab[len(vocab) + 1] = (('dropout'))
        if self.target_classes == 2:
            vocab[len(vocab) + 1] = (self.target_classes - 1, 'sigmoid')
        else:
            vocab[len(vocab) + 1] = (self.target_classes, 'softmax')
            
        return vocab

    def encode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        encoded_sequence = []
        for value in sequence:
            encoded_sequence.append(keys[values.index(value)])

        return encoded_sequence

    def decode_sequence(self, sequence):
        keys = list(self.vocab.keys())
        values = list(self.vocab.values())
        decoded_sequence = []
        for key in sequence:
            decoded_sequence.append(values[keys.index(key)])

        return decoded_sequence

class MLPGenerator(MLPSearchSpace):
    def __init__(self):
        self.target_classes = TARGET_CLASSES
        self.mlp_optimizer = MLP_OPTIMIZER
        self.mlp_lr = MLP_LEARNING_RATE
        self.mlp_decay = MLP_DECAY
        self.mlp_momentum = MLP_MOMENTUM
        self.mlp_dropout = MLP_DROPOUT
        self.mlp_loss_func = MLP_LOSS_FUNCTION
        self.mlp_one_shot = MLP_ONE_SHOT
        self.metrics = ['accuracy']

        super().__init__(TARGET_CLASSES)

        if self.mlp_one_shot:
            self.weights_file = 'LOGS/shared_weights.pkl'
            self.shared_weights = pd.DataFrame({'bigram_id': [], 'weights': []})
            if not os.path.exists(self.weights_file):
                print("Initializing shared weights dictionary...")
                self.shared_weights.to_pickle(self.weights_file)

    def create_model(self, sequence, mlp_input_shape):
        layer_configs = self.decode_sequence(sequence)
        model = Sequential()

        if len(mlp_input_shape) > 1:
            model.add_module('flatten', Flatten(mlp_input_shape))
            for i, layer_conf in enumerate(layer_configs):
                if layer_conf == 'dropout':
                    model.add_module('dropout', Dropout(self.mlp_dropout))
                else:
                    model.add_module('linear', Linear(units=layer_conf[0], activation=layer_conf[1]))
        else:
            for i, layer_conf in enumerate(layer_configs):
                if i == 0:
                    model.add_module('linear', Linear(units=layer_conf[0], activation=layer_conf[1], input_shape=mlp_input_shape))
                elif layer_conf == 'dropout':
                    model.add_module(Dropout(self.mlp_dropout, name='dropout'))
                else:
                    model.add_module('linear', Linear(units=layer_conf[0], activation=layer_conf[1]))
        return model

    def compile_model(self, model):
        if self.mlp_optimizer == 'sgd':
            optim = optimizers.SGD(model.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_decay, momentum=self.mlp_momentum)
        else:
            optim = getattr(optimizers, self.mlp_optimizer)(model.parameters(), lr=self.mlp_lr, weight_decay=self.mlp_decay)

        return optim

    def update_weights(self, model):
        layer_configs = ['input']
        for layer in model.layers:
            if 'flatten' in layer.name:
                layer_configs.append(('flatten'))
            elif 'dropout' not in layer.name:
                layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))
        
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        
        j = 0
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                bigram_ids = self.shared_weights['bigram_id'].values
                search_index = []
                for i in range(len(bigram_ids)):
                    if config_ids[j] == bigram_ids[i]:
                        search_index.append(i)
                if len(search_index) == 0:
                    self.shared_weights = self.shared_weights.append({'bigram_id': config_ids[j], 'weights': layer.get_weights()}, ignore_index=True)
                else:
                    self.shared_weights.at[search_index[0], 'weights'] = layer.get_weights()
                
                j += 1
        
        self.shared_weights.to_pickle(self.weights_file)

    def set_model_weights(self, model):
        layer_configs = ['input']
        for layer in model.layers:
            if 'flatten' in layer.name:
                layer_configs.append(('flatten'))
            elif 'dropout' not in layer.name:
                layer_configs.append((layer.get_config()['units'], layer.get_config()['activation']))
        
        config_ids = []
        for i in range(1, len(layer_configs)):
            config_ids.append((layer_configs[i - 1], layer_configs[i]))
        
        j = 0
        for i, layer in enumerate(model.layers):
            if 'dropout' not in layer.name:
                warnings.simplefilter(action='ignore', category=FutureWarning)
                bigram_ids = self.shared_weights['bigram_id'].values
                
                search_index = []
                for i in range(len(bigram_ids)):
                    if config_ids[j] == bigram_ids[i]:
                        search_index.append(i)
                if len(search_index) > 0:
                    print("Transferring weights for layer:", config_ids[j])
                    layer.set_weights(self.shared_weights['weights'].values[search_index[0]])
                
                j += 1

    def train_model(self, model, x_data, y_data, nb_epochs, validation_split=0.1, callbacks=None):
        if self.mlp_one_shot:
            self.set_model_weights(model)
            history = model.fit(x_data, y_data, epochs=nb_epochs, validation_split=validation_split, callbacks=callbacks, verbose=0)
            self.update_weights(model)
        else:
            history = model.fit(x_data, y_data, epochs=nb_epochs, validation_split=validation_split, callbacks=callbacks, verbose=0)

        return history
    
class Controller(MLPSearchSpace):
	def __init__(self):
		self.max_len = MAX_ARCHITECTURE_LENGTH
		self.controller_lstm_dim = CONTROLLER_LSTM_DIM
		self.controller_optimizer = CONTROLLER_OPTIMIZER
		self.controller_lr = CONTROLLER_LEARNING_RATE
		self.controller_decay = CONTROLLER_DECAY
		self.controller_momentum = CONTROLLER_MOMENTUM
		self.use_predictor = CONTROLLER_USE_PREDICTOR

		self.controller_weights = 'LOGS/controller_weights.h5'
		self.seq_data = []

		super().__init__(TARGET_CLASSES)

		self.controller_classes = len(self.vocab) + 1
        
	# Controller Architecture
	def control_model(self, controller_input_shape, controller_batch_size):
		main_input = torch.tensor(controller_batch_size, controller_input_shape) # name='main_input'
		x = LSTM(self.controller_lstm_dim)(main_input) # return_sequences=True
		main_output = Linear(self.controller_classes)(x) # activation='softmax', name='main_output'
		model = Model(inputs=[main_input], outputs=[main_output]) #???????????
		
		return model
	
	def hybrid_control_model(self, controller_input_shape, controller_batch_size):
		main_input = torch.tensor(controller_batch_size, controller_input_shape) # name='main_input'
		x = LSTM(self.controller_lstm_dim)(main_input) # return_sequences=True
		predictor_output = Linear(1)(x) # , activation='sigmoid', name='predictor_output'
		main_output = Linear(self.controller_classes)(x) # , activation='softmax', name='main_output'
		model = Model(inputs=[main_input], outputs=[main_output, predictor_output])
		
		return model
	
	def train_control_model(self, model, x_data, y_data, loss_func, controller_batch_size, nb_epochs):
		if self.controller_optimizer == 'sgd':
			optim = optimizers.SGD(lr=self.controller_lr, decay=self.controller_decay, momentum=self.controller_momentum, clipnorm=1.0)
		else:
			optim = getattr(optimizers, self.controller_optimizer)(lr=self.controller_lr, decay=self.controller_decay, clipnorm=1.0)
		model.compile(optimizer=optim, loss={'main_output': loss_func})
		if os.path.exists(self.controller_weights):
			model.load_weights(self.controller_weights)
		print("TRAINING CONTROLLER...")
		model.fit({'main_input': x_data},
					{'main_output': y_data.reshape(len(y_data), 1, self.controller_classes)},
					epochs=nb_epochs,
					batch_size=controller_batch_size,
					verbose=0)
		model.save_weights(self.controller_weights)

	def sample_architecture_sequences(self, model, number_of_samples):
		final_layer_id = len(self.vocab)
		dropout_id = final_layer_id - 1
		vocab_idx = [0] + list(self.vocab.keys())
		samples = []
		print("GENERATING ARCHITECTURE SAMPLES...")
		print('------------------------------------------------------')
		while len(samples) < number_of_samples:
			seed = []
			while len(seed) < self.max_len:
				sequence = pad_sequences([seed], maxlen=self.max_len - 1, padding='post')
				sequence = sequence.reshape(1, 1, self.max_len - 1)
				if self.use_predictor:
					(probab, _) = model.predict(sequence)
				else:
					probab = model.predict(sequence)
				probab = probab[0][0]
				next = np.random.choice(vocab_idx, size=1, p=probab)[0]
				if next == dropout_id and len(seed) == 0:
					continue
				if next == final_layer_id and len(seed) == 0:
					continue
				if next == final_layer_id:
					seed.append(next)
					break
				if len(seed) == self.max_len - 1:
					seed.append(final_layer_id)
					break
				if not next == 0:
					seed.append(next)
			if seed not in self.seq_data:
				samples.append(seed)
				self.seq_data.append(seed)
		return samples
	
	def get_predicted_accuracies_hybrid_model(self, model, seqs):
		pred_accuracies = []
		for seq in seqs:
			control_sequences = pad_sequences([seq], maxlen=self.max_len, padding='post')
			xc = control_sequences[:, :-1].reshape(len(control_sequences), 1, self.max_len - 1)
			(_, pred_accuracy) = [x[0][0] for x in model.predict(xc)]
			pred_accuracies.append(pred_accuracy[0])
		return pred_accuracies

# utils
def clean_log():
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            os.remove('LOGS/{}'.format(file))

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def log_event():
    dest = 'LOGS'
    while os.path.exists(dest):
        dest = 'LOGS/event{}'.format(np.random.randint(10000))
    os.mkdir(dest)
    filelist = os.listdir('LOGS')
    for file in filelist:
        if os.path.isfile('LOGS/{}'.format(file)):
            shutil.move('LOGS/{}'.format(file),dest)

# utils
def get_latest_event_id():
    all_subdirs = ['LOGS/' + d for d in os.listdir('LOGS') if os.path.isdir('LOGS/' + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return int(latest_subdir.replace('LOGS/event', ''))

def load_nas_data():
    event = get_latest_event_id()
    data_file = 'LOGS/event{}/nas_data.pkl'.format(event)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data

def sort_search_data(nas_data):
    val_accs = [item[1] for item in nas_data]
    sorted_idx = np.argsort(val_accs)[::-1]
    nas_data = [nas_data[x] for x in sorted_idx]
    return nas_data

def get_top_n_architectures(n):
    data = load_nas_data()
    data = sort_search_data(data)
    search_space = MLPSearchSpace(TARGET_CLASSES)
    print('Top {} Architectures:'.format(n))
    for seq_data in data[:n]:
        print('Architecture', search_space.decode_sequence(seq_data[0]))
        print('Validation Accuracy:', seq_data[1])

class MLPNAS(Controller):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.target_classes = TARGET_CLASSES
        self.controller_sampling_epochs = CONTROLLER_SAMPLING_EPOCHS
        self.samples_per_controller_epoch = SAMPLES_PER_CONTROLLER_EPOCH
        self.controller_train_epochs = CONTROLLER_TRAINING_EPOCHS
        self.architecture_train_epochs = ARCHITECTURE_TRAINING_EPOCHS
        self.controller_loss_alpha = CONTROLLER_LOSS_ALPHA

        self.data = []
        self.nas_data_log = 'LOGS/nas_data.pkl'
        clean_log()

        super().__init__()

        self.model_generator = MLPGenerator()

        self.controller_batch_size = len(self.data)
        self.controller_input_shape = (1, MAX_ARCHITECTURE_LENGTH - 1)
        print(self.controller_batch_size)
        print(self.controller_input_shape)

        if self.use_predictor:
            self.controller_model = self.hybrid_control_model(self.controller_input_shape, self.controller_batch_size)
        else:
            self.controller_model = self.control_model(self.controller_input_shape, self.controller_batch_size)

    # Training MLP models
    def create_architecture(self, sequence):
        if self.target_classes == 2:
            self.model_generator.loss_func = 'binary_crossentropy'
        model = self.model_generator.create_model(sequence, np.shape(self.x[0]))
        model = self.model_generator.compile_model(model)
        return model
    
    def train_architecture(self, model):
        x, y = unison_shuffled_copies(self.x, self.y)
        history = self.model_generator.train_model(model, x, y, self.architecture_train_epochs)
        return history
    
    # Storing the training metrics
    def append_model_metrics(self, sequence, history, pred_accuracy=None):
        if len(history.history['val_accuracy']) == 1:
            if pred_accuracy:
                self.data.append([sequence,
                                    history.history['val_accuracy'][0],
                                    pred_accuracy])
            else:
                self.data.append([sequence,
                                    history.history['val_accuracy'][0]])
            print('validation accuracy: ', history.history['val_accuracy'][0])
        else:
            val_acc = np.ma.average(history.history['val_accuracy'],
                                    weights=np.arange(1, len(history.history['val_accuracy']) + 1),
                                    axis=-1)
            if pred_accuracy:
                self.data.append([sequence,
                                    val_acc,
                                    pred_accuracy])
            else:
                self.data.append([sequence,
                                    val_acc])
            print('validation accuracy: ', val_acc)
    
    # Preparing data for controller
    def prepare_controller_data(self, sequences):
        controller_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        xc = controller_sequences[:, :-1].reshape(len(controller_sequences), 1, self.max_len - 1)
        yc = to_categorical(controller_sequences[:, -1], self.controller_classes)
        val_acc_target = [item[1] for item in self.data]
        return xc, yc, val_acc_target
    
    # Implementing REINFORCE Gradient
    def get_discounted_reward(self, rewards):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        for t in range(len(rewards)):
            running_add = 0.
            exp = 0.
            for r in rewards[t:]:
                running_add += self.controller_loss_alpha**exp * r
                exp += 1
            discounted_r[t] = running_add
        discounted_r = (discounted_r - discounted_r.mean()) / discounted_r.std()
        return discounted_r

    def custom_loss(self, target, output):
        baseline = 0.5
        reward = np.array([item[1] - baseline for item in self.data[-self.samples_per_controller_epoch:]]).reshape(
            self.samples_per_controller_epoch, 1)
        discounted_reward = self.get_discounted_reward(reward)
        loss = - K.log(output) * discounted_reward[:, None]
        return loss
    
    # Training the controller
    def train_controller(self, model, x, y, pred_accuracy=None):
        if self.use_predictor:
            self.train_hybrid_model(model,
                                    x,
                                    y,
                                    pred_accuracy,
                                    self.custom_loss,
                                    len(self.data),
                                    self.controller_train_epochs)
        else:
            self.train_control_model(model,
                                     x,
                                     y,
                                     self.custom_loss,
                                     len(self.data),
                                     self.controller_train_epochs)

    # The Main NAS loop
    def search(self):
        for controller_epoch in range(self.controller_sampling_epochs):
            print('------------------------------------------------------------------')
            print('                       CONTROLLER EPOCH: {}'.format(controller_epoch))
            print('------------------------------------------------------------------')
            sequences = self.sample_architecture_sequences(self.controller_model, self.samples_per_controller_epoch)
            if self.use_predictor:
                pred_accuracies = self.get_predicted_accuracies_hybrid_model(self.controller_model, sequences)
            for i, sequence in enumerate(sequences):
                print('Architecture: ', self.decode_sequence(sequence))
                model = self.create_architecture(sequence)
                history = self.train_architecture(model)
                if self.use_predictor:
                    self.append_model_metrics(sequence, history, pred_accuracies[i])
                else:
                    self.append_model_metrics(sequence, history)
                print('------------------------------------------------------')
            xc, yc, val_acc_target = self.prepare_controller_data(sequences)
            self.train_controller(self.controller_model,
                                    xc,
                                    yc,
                                    val_acc_target[-self.samples_per_controller_epoch:])
        with open(self.nas_data_log, 'wb') as f:
            pickle.dump(self.data, f)
        log_event()
        
        return self.data
    
data = pd.read_csv('DATASETS/wine-quality.csv')
x = data.drop('quality_label', axis=1, inplace=False).values
y = pd.get_dummies(data['quality_label']).values

# x : (4898, 13)
# y : (4898, 3)

nas_object = MLPNAS(x, y)
data = nas_object.search()

get_top_n_architectures(TOP_N)