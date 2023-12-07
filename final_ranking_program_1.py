import pandas as pd
import numpy as np # linear algebra
from tqdm import tqdm
import tensorflow as tf
import transformers
from tokenizers import BertWordPieceTokenizer
import json
import warnings
import xgboost as xgb


warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

raw = pd.read_csv('review_final_test.csv')
raw = raw.iloc[:,1:]


# useful model

# Configuration
EPOCHS = 3
BATCH_SIZE = 12
MAX_LEN = 129

model = tf.keras.models.load_model(r'saved_model/nlp_review_10k_epoch_3_useful_balanced',
                                       custom_objects={"TFDistilBertModel":transformers.TFDistilBertModel})
print(model.summary())

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

pred_res= model.predict(predict_dataset,verbose=1)
res_df = pd.DataFrame(pred_res,columns=['score'])
res_df['useful_flag'] = (res_df['score']>=0.5).astype(int)

raw['nlp_predicted_useful'] = res_df['useful_flag']


# regression model

# Process
def df_user_process(df_review):
    # return df_user containing all users existing in df_review
    user_path = r'E:\dev\yelp_academic_dataset_user.json'  # modify the user.json path ***
    selected_users = set(df_review['user_id'])
    selected_lines = []

    with open(user_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i % 10000 == 0:
                print(i)
            data = json.loads(line)
            if data['user_id'] in selected_users:
                selected_lines.append(line)

    data = [json.loads(line) for line in selected_lines]
    df_user = pd.DataFrame(data)

    df_user.rename(columns={'useful': 'history_useful',
                            'funny': 'history_funny',
                            'cool': 'history_cool'}, inplace=True)

    df_user['elite'] = df_user['elite'].apply(lambda x: len(x.split(',')) if x else 0)
    df_user['friends'] = df_user['friends'].apply(lambda x: len(x.split(',')) if x != 'None' else 0)

    df_review['date'] = pd.to_datetime(df_review['date'])
    latest_date = max(df_review['date'])
    df_user['yelping_since'] = pd.to_datetime(df_user['yelping_since'])
    df_user['yelping_since'] = (latest_date - df_user['yelping_since']).dt.days // 365

    df_user = df_user.drop(['compliment_profile', 'compliment_cute', 'compliment_list', 'compliment_funny'], axis=1)
    return df_user


def df_comb_process(df_review, df_user):
    # return df_combined for train/test/val (Y - useful; X- others)
    df_combined = pd.merge(df_review[['user_id', 'useful']], df_user, on='user_id', how='inner')
    df_combined = df_combined.drop(['name'], axis=1)
    return df_combined

df_review = raw

df_user = df_user_process(df_review)
df_comb = df_comb_process(df_review, df_user)
X = df_comb.drop(columns=['user_id', 'useful'])
Y = df_comb['useful']

model_path = 'saved_model/xgb_regression_user_25k_useful.model'

xgb_model = xgb.Booster()
xgb_model.load_model(model_path)
print("XGBoost model loaded successfully.")

raw['xgb_predicted_useful'] = xgb_model.predict(xgb.DMatrix(X))

raw.to_csv('rank.csv', index=False)

