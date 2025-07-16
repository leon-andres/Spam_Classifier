
import pandas as pd

def open_data():
    df = pd.read_csv("./data/enron_spam_data.csv")

    df['text'] = df['Subject'].fillna('') + ' ' + df['Message'].fillna('')

    df = df[['text','Spam/Ham']]
    df['message_length'] = df['text'].apply(len)

    df['Label'] = df['Spam/Ham'].map(
        {
            'ham':0,
            'spam':1
        }
    )

    return df

def main():
    df = open_data()
    print(df.head())

if __name__ == "__main__":
    main()