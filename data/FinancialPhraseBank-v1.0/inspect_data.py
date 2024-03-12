import pickle


with open('data/FinancialPhraseBank-v1.0/dataset_zh_split50.pkl', 'rb') as f:
    dataset = pickle.load(f)


def label2zh(lab: str) -> str:
    lab = lab.strip()
    if lab.lower() == 'positive':
        return '正面'
    elif lab.lower() == 'negative':
        return '负面'
    elif lab.lower() == 'neutral':
        return '中立'
    else:
        raise ValueError(f'Unrecognized label type: {lab.lower()}.')


if __name__ == '__main__':
    for i in range(len(dataset['sentence'])):
        input("输入回车以查看下一个样本...")
        print(f"第{i}个样本：{dataset['sentence'][i]}"
              f"请给出上述文本的情感分类（正面、负面或中立）：{label2zh(dataset['label'][i])}")
