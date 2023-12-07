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

#IMP DATA FOR CONFIG

AUTO = tf.data.experimental.AUTOTUNE

# Configuration
EPOCHS = 3
BATCH_SIZE = 12
MAX_LEN = 129


# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, False)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


df = pd.read_csv('review_useful_data.csv')
df['useful_flag'] = (df['useful']>=10).astype(int)

df_test_2 = pd.read_csv('review_useful_data_0_to_10.csv')
df_test_2['useful_flag'] = (df_test_2['useful']>=10).astype(int)

train_set_x, test_set_x, train_set_y, test_set_y = (
    train_test_split(df['text'],df['useful_flag'],
                     test_size=0.25, random_state=123))

train_x,valid_x,train_y,valid_y = (
    train_test_split(train_set_x,train_set_y,test_size=0.25, random_state=123))

test_set_x = pd.concat([test_set_x,df_test_2['text']])
test_set_y = pd.concat([test_set_y,df_test_2['useful_flag']])

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
x_train = fast_encode(train_x.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_valid = fast_encode(valid_x.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_test = fast_encode(test_set_x.astype(str), fast_tokenizer, maxlen=MAX_LEN)

y_train = train_y.values
y_valid = valid_y.values
y_test = test_set_y.values

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_test,y_test))
    .batch(BATCH_SIZE)
)


def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


transformer_layer = (
    transformers.TFDistilBertModel
    .from_pretrained('distilbert-base-multilingual-cased')
)
model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()

n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

n_steps = x_valid.shape[0] // BATCH_SIZE
train_history_2 = model.fit(
    valid_dataset.repeat(),
    steps_per_epoch=n_steps,
    epochs=EPOCHS*2
)

model_path = 'saved_model/nlp_review_10k_epoch_3_useful_balanced'
model.save(model_path,save_format="h5")
print('model is saved at: ', model_path)


# accuracy for classification
loss, accuracy = model.evaluate(test_dataset)
print(f'test Loss:{loss}')
print(f'test Accuracy:{accuracy}')
