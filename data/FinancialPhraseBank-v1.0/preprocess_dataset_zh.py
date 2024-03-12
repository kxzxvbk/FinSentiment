import argparse
import pickle
from tqdm import tqdm
import requests
import random
import json
from hashlib import md5

# Set your own appid/appkey.
# This id is only for personal use !!!
appid = '20240312001991276'
appkey = 'XMP7uDhLwh9ls31MIp2X'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'en'
to_lang = 'zh'

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

def translate(query):
    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()

    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    return eval(json.dumps(result, indent=4, ensure_ascii=False))['trans_result'][0]['dst']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='data/FinancialPhraseBank-v1.0/dataset_en_split50.pkl')
    parser.add_argument('--outdir', type=str, default='data/FinancialPhraseBank-v1.0/dataset_zh_split50.pkl')
    args = parser.parse_args()

    with open(args.indir, 'rb') as f:
        dataset_en = pickle.load(f)

    zh_sentences = []
    for i in tqdm(range(len(dataset_en['sentence']))):
        q = dataset_en['sentence'][i]
        zh_sentences.append(translate(q))
        with open('translation.txt', 'a+', encoding='utf8') as f:
            f.write(zh_sentences[-1] + '\n')

    dataset_zh = {
        'sentence': zh_sentences,
        'label': dataset_en['label']
    }

    with open(args.outdir, 'wb') as f:
        pickle.dump(dataset_zh, f)
