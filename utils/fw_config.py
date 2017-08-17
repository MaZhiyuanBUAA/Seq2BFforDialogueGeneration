import tensorflow as tf

TEST_DATASET_PATH = '/home/zyma/work/models/evaluation/std_test.post'
#SAVE_DATA_DIR = '/home/easemob/workspace/chatbot1/'

data_dir = '/home/zyma/work/data_daily_punct'
model_dir = 'fw'
results_dir = 'results'
emb_path ='fw/embeddings.pkl'#'Embeddings'

learning_rate = 0.5 #'Learning rate.'
learning_rate_decay_factor = 0.99 #'Learning rate decays by this much.')
max_gradient_norm = 5.0 # 'Clip gradients to this norm.')
batch_size = 128 # 'Batch size to use during training.')

vocab_size = 110000 # 'Dialog vocabulary size.')#total20003
size = 128 # 'Size of each model layer.')
num_layers = 3 # 'Number of layers in the model.')
embedding_dim = 128 #'dimention of embeddings')
max_train_data_size = 1000000 # 'Limit on the size of training data (0: no limit).')
steps_per_checkpoint = 100 # 'How many training steps to do per checkpoint.')

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#BUCKETS = [(14+3*i, 14+3*i) for i in range(5)]+[(44,44)]#chatbot1
#BUCKETS = [(10, 15), (20, 25),(40,50)]#nn_models and char_models
BUCKETS = [(5, 15), (10, 20)]#log_models

#global global_memory,global_output
#global_memory = {'inp':None,'attns':None,'state':None}
#global_output = {'output':None,'attns':None,'state':None}

