#coding:utf-8
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tensorflow as tf
import pickle
from config import TEST_DATASET_PATH, FLAGS
import data_utils
from seq2seq_model_utils import create_model,get_predicted_sentence
from toolkits import sentencePMI

def predict():
   
    f = open('pkl_tianya/q_table.pkl')
    qtable = pickle.load(f)
    f.close()
    #f = open('pkl_file/n_table.pkl')
    #ntable = pickle.load(f)
    #f.close()
    f = open('pkl_tianya/co_table.pkl')
    cotable = pickle.load(f)
    f.close()
    f = open('/home/zyma/work/data_daily_punct/nouns2500.in')
    nouns = f.readlines()
    nouns = [ele.strip() for ele in nouns]
    f.close()
    
    def _get_test_dataset():
        with open(TEST_DATASET_PATH) as test_fh:
            test_sentences = [s.strip() for s in test_fh.readlines()]
        return test_sentences
    results_filename = '_'.join(['results', str(FLAGS.num_layers), str(FLAGS.size), str(FLAGS.vocab_size)])
    results_path = os.path.join(FLAGS.results_dir, results_filename)
    #ss = u'你好'
    #ss = ss.encode('utf-8')
    #print(ss)
    with tf.Session() as sess, open(results_path, 'a') as results_fh:
    #with tf.Session() as sess:
        # Create model and load parameters.
        bw_model,fw_model = create_model(sess)
        bw_model.batch_size = 1
        fw_model.batch_size = 1
        # Load vocabularies.
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
        print(vocab.items()[:20])

        test_dataset = _get_test_dataset()
        #test_dataset = test_dataset[374:]
        #predicted_sentences = beam_search(test_dataset,vocab,rev_vocab,model,sess)
        #results_fh.write('\n'.join(predicted_sentences))

        for sentence in test_dataset:
            # Get token-ids for the input sentence.
            #best,predicted_sentences,scores = beam_search(sentence, vocab, rev_vocab, model, sess)
            key_word = sentencePMI(sentence,cotable,qtable,nouns)
            print('key_word:%s'%key_word)
            bw_sentence = get_predicted_sentence(sentence,key_word, vocab, rev_vocab, bw_model, sess)
            print(bw_sentence)
            bw_sentence = bw_sentence.split()[:10]
            bw_sentence.reverse()
            bw_sentence = ' '.join(bw_sentence)
            print('bw_sentence:%s'%bw_sentence)
            predicted_sentences = get_predicted_sentence(sentence,bw_sentence, vocab,rev_vocab,fw_model,sess)
            print (sentence+' -> '+predicted_sentences)
            #predicted_sentences = predicted_sentences.split()
            #predicted_sentences = ' '.join(predicted_sentences[:1])
            #predicted_sentences = get_predicted_sentence(sentence,None,vocab,rev_vocab,fw_model,sess)
            #print(sentence+' ---> '+predicted_sentences)
            #print ('\n'.join([str(ele)+','+predicted_sentences[ind] for ind,ele in enumerate(scores)]))
            #print(len(scores))
            #results_fh.write(best+'\n')

            results_fh.write(predicted_sentences+'(%s)'%key_word+'\n')

if __name__=='__main__':
  predict()
