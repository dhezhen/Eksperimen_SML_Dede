# automate_DedeHusen.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os


def load_data(file_path):
    """Membaca dataset dari file"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Melakukan semua tahapan preprocessing"""
    df = df.copy()

    #  menghapus kolom yang tidak relevan
    df.drop(columns=['student_id', 'age'], inplace=True, errors='ignore')

    # Mengisi missing value
    df.ffill(inplace=True)

    # Label Encoding untuk kolom kategorikal (misal: gender, status)
    label_cols = df.select_dtypes(include='object').columns
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df

def split_data(df, target_column='exam_score', test_size=0.2, random_state=42):
    """Membagi data menjadi train dan test"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

import argparse

# pemanggilan fungsi
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess student dataset and split into train/test sets.")
    parser.add_argument('--data', type=str, default='../Eksperimen_SML_Dede/student_habits_performance.csv', help='Path to the dataset CSV file')
    parser.add_argument('--target', type=str, default='exam_score', help='Target column name (ubah sesuai dataset)')
    args = parser.parse_args()

    data = load_data(args.data)
    preprocessed = preprocess_data(data)
    if args.target not in preprocessed.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset columns: {preprocessed.columns.tolist()}")
    X_train, X_test, y_train, y_test = split_data(preprocessed, target_column=args.target)
    # Simpan hasil preprocessing
    preprocessed.to_csv('student_habit_performance_preprocessing.csv', index=False)
    print("Data preprocessing selesai. Data telah disimpan di 'student_habit_performance_preprocessing.csv'.")