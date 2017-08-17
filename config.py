import tensorflow as tf

TEST_DATASET_PATH = '/home/zyma/work/models/evaluation/test1k.query'
#SAVE_DATA_DIR = '/home/easemob/workspace/chatbot1/'

tf.app.flags.DEFINE_string('data_dir', '/home/zyma/work/data_daily_punct', 'Data directory')
tf.app.flags.DEFINE_string('model_dir', 'fw_model/nn_models', 'Train directory')
tf.app.flags.DEFINE_string('results_dir', 'results', 'Train directory')
tf.app.flags.DEFINE_string('emb_path','nn_models/embeddings.pkl','Embeddings')

tf.app.flags.DEFINE_float('learning_rate', 0.5, 'Learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.99, 'Learning rate decays by this much.')
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size to use during training.')

tf.app.flags.DEFINE_integer('vocab_size', 110000, 'Dialog vocabulary size.')#total20003
tf.app.flags.DEFINE_integer('size', 128, 'Size of each model layer.')
tf.app.flags.DEFINE_integer('num_layers', 3, 'Number of layers in the model.')
tf.app.flags.DEFINE_integer('embedding_dim',128,'dimention of embeddings')
tf.app.flags.DEFINE_integer('max_train_data_size', 1000000, 'Limit on the size of training data (0: no limit).')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'How many training steps to do per checkpoint.')

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#BUCKETS = [(14+3*i, 14+3*i) for i in range(5)]+[(44,44)]#chatbot1
#BUCKETS = [(10, 15), (20, 25),(40,50)]#nn_models and char_models
BUCKETS = [(5, 15), (10, 20)]#log_models

#global global_memory,global_output
#global_memory = {'inp':None,'attns':None,'state':None}
#global_output = {'output':None,'attns':None,'state':None}

