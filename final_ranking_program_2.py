import pandas as pd
import numpy as np # linear algebra
from tqdm import tqdm
import tensorflow as tf
import transformers
from tokenizers import BertWordPieceTokenizer
import json
import warnings
import xgboost as xgb

# star rate model
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

raw = pd.read_csv('rank.csv')


# useful model

# Configuration
EPOCHS = 10
BATCH_SIZE = 8
MAX_LEN = 192


predict_x = raw['text']

def load_tokenizer():
    # First load the real tokenizer
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    # Save the loaded tokenizer locally
    tokenizer.save_pretrained('.')
    # Reload it with the huggingface tokenizers library
    fast_tokenizer_res = BertWordPieceTokenizer('vocab.txt', lowercase=False)
    return fast_tokenizer_res

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(length=maxlen)
    all_ids = []

    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i + chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)


fast_tokenizer = load_tokenizer()
x_predict = fast_encode(predict_x.astype(str), fast_tokenizer, maxlen=MAX_LEN)

predict_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_predict)
    .batch(BATCH_SIZE)
)


model = tf.keras.models.load_model(r'saved_model/nlp_review_10k_epoch_10',
                                       custom_objects={"TFDistilBertModel":transformers.TFDistilBertModel})
print(model.summary())

fast_tokenizer = load_tokenizer()
x_predict = fast_encode(predict_x.astype(str), fast_tokenizer, maxlen=MAX_LEN)

predict_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_predict)
    .batch(BATCH_SIZE)
)


pred_res= model.predict(predict_dataset,verbose=1)
res_df = pd.DataFrame(pred_res)
weights = list(range(1, len(res_df.columns)+1))
raw['nlp_predicted_star'] = (res_df*weights).sum(axis=1)/res_df.sum(axis=1)

def calculate_final_score(row):
    if row['nlp_predicted_useful']==0 or abs(row['nlp_predicted_star']-row['stars'])>1:
        return 0
    else:
        return row['xgb_predicted_useful']

raw['final_score'] = raw.apply(calculate_final_score,axis=1)

raw_sorted = raw.sort_values(by=['stars','final_score'],ascending=[False,False])

raw_sorted = raw_sorted[['review_id','user_id','business_id','text','stars']]

raw_sorted.to_csv('rank.csv')