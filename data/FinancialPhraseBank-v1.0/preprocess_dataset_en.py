import argparse
import pickle


def get_dataset(file_path):
    res = []
    with open(file_path, encoding="iso-8859-1") as f:
        for id_, line in enumerate(f):
            sentence, label = line.rsplit("@", 1)
            res.append({'sentence': sentence, 'label': label})

    return {
        'sentence': [item['sentence'] for item in res],
        'label': [item['label'] for item in res]
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='50')
    parser.add_argument('--outdir', type=str, default='data/FinancialPhraseBank-v1.0/dataset_en_split50.pkl')
    args = parser.parse_args()

    split2path = {
        "50": "./data/FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
        "66": "./data/FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
        "75": "./data/FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
        "All": "./data/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
    }

    dataset = get_dataset(split2path[args.split])

    with open(args.outdir, 'wb') as f:
        pickle.dump(dataset, f)
