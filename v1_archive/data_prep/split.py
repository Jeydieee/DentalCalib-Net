import json
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

def load_labels(path='strat_labels.json'):
    with open(path) as f:
        data = json.load(f)
    return data['image_ids'], data['labels']

def stratified_split(image_ids, y, seed=42):
    train_ids, temp_ids, y_train, y_temp = train_test_split(
        image_ids, y, test_size=0.40, stratify=y, random_state=seed
    )
    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )
    return (train_ids, y_train), (val_ids, y_val), (test_ids, y_test)

def verify_split(train, val, test):
    train_ids, y_train = train
    val_ids, y_val = val
    test_ids, y_test = test

    print("train:", Counter(y_train))
    print("val:  ", Counter(y_val))
    print("test: ", Counter(y_test))

    assert not (set(train_ids) & set(val_ids)), "train/val overlap"
    assert not (set(train_ids) & set(test_ids)), "train/test overlap"
    assert not (set(val_ids) & set(test_ids)), "val/test overlap"

def save_splits(train, val, test):
    pd.DataFrame({'image_id': train[0]}).to_csv('train_ids.csv', index=False)
    pd.DataFrame({'image_id': val[0]}).to_csv('val_ids.csv', index=False)
    pd.DataFrame({'image_id': test[0]}).to_csv('test_ids.csv', index=False)
    print("Saved train_ids.csv, val_ids.csv, test_ids.csv")

def main():
    image_ids, y = load_labels()
    train, val, test = stratified_split(image_ids, y)
    verify_split(train, val, test)
    save_splits(train, val, test)

if __name__ == "__main__":
    main()