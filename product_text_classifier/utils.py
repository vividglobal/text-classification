import json
import pickle


def load_json(fn):
    with open(fn, mode='r', encoding='utf8') as f:
        data = json.load(f)
    return data


def dump_json(data, fn, indent=None,):
    with open(fn, mode='w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_jsonl(fn):
    data = []
    with open(fn, mode='r', encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_tokens(item):
    tokens = []
    for box in item['result']:
        text = box['text']
        if text:
            tokens.append(text)
    return tokens


def dump_jsonl(data, fn):
    with open(fn, mode='w', encoding='utf8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False,))
            f.write('\n')


def load_pickle(fn):
    with open(fn, mode='rb') as f:
        data = pickle.load(f)
    return data


def dump_pickle(data, fn):
    with open(fn, mode='wb') as f:
        pickle.dump(data, f)


