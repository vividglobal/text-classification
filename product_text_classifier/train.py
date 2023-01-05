import pandas as pd
import random
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.linear_model import SGDClassifier
from argparse import ArgumentParser

from product_text_classifier.utils import dump_pickle, dump_json
from product_text_classifier.models import Analyzer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_fn', type=str, default='product_text_classifier/data/samples/samples.csv',)
    parser.add_argument('--seed', type=int, default=42,)
    parser.add_argument('--brand_out_fn', type=str, default='product_text_classifier/data/checkpoints/product_classifier_level1.pkl')
    parser.add_argument('--step_out_dir', type=str, default='product_text_classifier/data/checkpoints/brands')

    args = parser.parse_args()
    return args


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def split_data(
        df: pd.DataFrame,
        test_size=0.1,
):
    print('labels:', df[f'label1'].unique())

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df[f'label1'],
    )

    return train_df, test_df


def train_brand_classification(
        train_df,
        test_df,
        out_fn,
):
    train_docs = train_df['text'].astype(str)
    train_targets = train_df[f'label1'].astype(str)
    test_docs = test_df['text'].astype(str)
    test_targets = test_df[f'label1'].astype(str)
    text_clf = Pipeline(
        [('vect', CountVectorizer(
            lowercase=True,
            analyzer=Analyzer(n=3),
            min_df=10,
            max_df=0.9,
        )),
         ('clf', SGDClassifier(
             loss='hinge',
             penalty='l2',
             alpha=1e-3,
             random_state=42,
             max_iter=50,
             tol=None,
             class_weight='balanced',
         )), ],
    )

    text_clf.fit(X=train_docs, y=train_targets)

    preds = text_clf.predict(test_docs)

    print(f'prec = {precision_score(y_true=test_targets, y_pred=preds, average="macro"):.4f}')
    print(f'rec = {recall_score(y_true=test_targets, y_pred=preds, average="macro"):.4f}')
    print(f'f1 = {f1_score(y_true=test_targets, y_pred=preds, average="macro"):.4f}')

    dump_pickle(text_clf, out_fn)


def train_step_classification(
        train_df,
        test_df,
        out_dir,
):
    brands = [b for b in train_df['label1'].unique().tolist() if b not in ['no brand', 'f99foods', 'heinz']]

    train_df['lens'] = train_df['text'].apply(lambda x: len(x.split()))
    test_df['lens'] = test_df['text'].apply(lambda x: len(x.split()))

    train_df = train_df[train_df['lens'] > 1]
    test_df = test_df[test_df['lens'] > 1]
    print('num train samples:', len(train_df))
    print('num test samples:', len(test_df))

    all_test_preds = []
    all_test_y = []
    unique_brands = []

    for brand in brands:
        brand_train_df = train_df[train_df['label1'] == brand]
        if brand_train_df['label3'].nunique() == 1:
            unique_brands.append(brand)
            continue

        brand_test_df = test_df[test_df['label1'] == brand]

        brand_text_clf = Pipeline(
            [('vect', CountVectorizer(
                lowercase=True,
                analyzer=Analyzer(n=6),
                min_df=5,
                max_df=0.9,
            )),
             ('clf', SGDClassifier(
                 loss='hinge',
                 penalty='l2',
                 alpha=1e-3,
                 random_state=42,
                 max_iter=100,
                 tol=None,
                 class_weight='balanced',
             )), ],
        )
        brand_text_clf.fit(X=brand_train_df['text'], y=brand_train_df['label3'])
        preds = brand_text_clf.predict(brand_test_df['text'])

        brand_f1 = f1_score(y_true=brand_test_df['label3'], y_pred=preds, average='macro')
        print(f'{brand}: f1 = {brand_f1:.4%}')
        all_test_preds.extend(preds)
        all_test_y.extend(brand_test_df['label3'])

        dump_pickle(brand_text_clf, os.path.join(out_dir, f'product_classifier_level3_{brand}.pkl'))

    # Dump mapping
    mapping = {
        brand: f'product_classifier_level3_{brand}.pkl'
        for brand in brands if brand not in ['bubs australia', 'megmilksnowbrand']
    }

    dump_json(mapping, os.path.join(out_dir, 'mapping.json'), indent=4, )

    # Evaluate all
    all_f1 = f1_score(
        y_true=all_test_y,
        y_pred=all_test_preds,
        average='macro'
    )

    print('#' * 10)
    print(f'All f1 = {all_f1:.4%}')


if __name__ == '__main__':
    args = parse_args()
    seed_all(seed=args.seed,)

    df = pd.read_csv(args.data_fn, sep='\t')
    train_df, test_df = split_data(df, test_size=0.3)
    train_df['text'] = train_df['text'].astype(str)
    test_df['text'] = test_df['text'].astype(str)

    train_brand_classification(
        train_df=train_df,
        test_df=test_df,
        out_fn=args.brand_out_fn,
    )

    train_step_classification(
        train_df=train_df,
        test_df=test_df,
        out_dir=args.step_out_dir,
    )
