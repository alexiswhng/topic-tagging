from datetime import date
import feedparser as fp
from dateutil import parser
import ssl
import pickle

ssl._create_default_https_context = ssl._create_unverified_context

from app import app, db, Article

app.app_context().push()
db.drop_all()
db.create_all()

relevant = pickle.load(open('relevancy.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

model = pickle.load(open('model.pkl', 'rb'))

def relevancy(text):
    text = str(text)
    tfidf_text = vectorizer.transform([text])
    pred = relevant.predict(tfidf_text)
    return pred.item()

import pickle
import sys 
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F


def pred_tags(text):
    
    new_text = [str(text)]

    tokenizer_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    labels = ['5G', 'AI', 'Award', 'Banking as a Service', 'Biometrics',
           'Blockchain', 'CBDC', 'Cloud', 'Crypto', 'Cybersecurity',
           'Decentralized ID', 'DeFi', 'Digital', 'Digital ID',
           'Digital Lending', 'Embedded Banking', 'Fines', 'FinTech',
           'Tech Innovation', 'Metaverse', 'NFTs', 'Open Banking',
           'Passwordless authentication', 'Payments', 'Quantum',
           'Real-Time Payments', 'Sustainability', 'Sustainable Finance',
           'Tech Talent', 'Tech Spend']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the new text
    encoded_new_text = tokenizer(
        new_text,
        add_special_tokens=True,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    outputs = model(input_ids=encoded_new_text['input_ids'].to(device), 
                    attention_mask=encoded_new_text['attention_mask'].to(device))
    
    probs = F.sigmoid(outputs.logits)
    predictions = (probs > 0.9).squeeze().long()

    tags = list()

    for i in range(len(predictions)):
        if predictions[i] == 1:
            tags.append(labels[i])

    return tags


sites = ["https://www.finextra.com/rss/headlines.aspx", 
         "https://www.coindesk.com/arc/outboundfeeds/rss/"]


for i in sites:
    feed = fp.parse(i)

    article_title = feed.channel.title

    for item in feed.entries:
        result = relevancy(item.title)
        if result == 'relevant':
            print(item.title)

            labels = pred_tags(item.title)


            article = Article(
                title=item.title,
                description=item.description,
                pub_date=parser.parse(item.published),
                link=item.link,
                source=article_title,
                tag = ', '.join(labels)
            )

            db.session.add(article)
            db.session.commit()



