import pandas as pd
import numpy as np # linear algebra
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import transformers
from tokenizers import BertWordPieceTokenizer
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Configuration
EPOCHS = 3
BATCH_SIZE = 12
MAX_LEN = 129



model = tf.keras.models.load_model(r'saved_model/nlp_review_10k_epoch_3_useful_balanced',
                                       custom_objects={"TFDistilBertModel":transformers.TFDistilBertModel})
print(model.summary())

predict_path = 'E:/dev/review.csv'
df = pd.read_csv(predict_path,nrows=100)
predict_x = df['text']


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

pred_res= model.predict(predict_dataset,verbose=1)
res_df = pd.DataFrame(pred_res,columns=['score'])
res_df['useful_flag'] = (res_df['score']>=0.5).astype(int)
df['predicted_useful'] = res_df['useful_flag']

output_path = 'predict_useful_res.csv'
df.to_csv(output_path,index=False)
