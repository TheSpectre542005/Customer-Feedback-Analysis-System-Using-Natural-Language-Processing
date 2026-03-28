
import os, re, pickle, torch, nltk, streamlit as st
import torch.nn as nn
from scipy.sparse import hstack
from nltk.corpus  import stopwords
from nltk.stem    import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)

STOP_WORDS   = set(stopwords.words('english'))
LEMMATIZER   = WordNetLemmatizer()
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IDX_TO_LABEL = {0:'Positive', 1:'Neutral', 2:'Negative'}
NUM_CLASSES  = 3

CFG = {
    'Positive': {'emoji':'😊','color':'#2ecc71'},
    'Neutral':  {'emoji':'😐','color':'#95a5a6'},
    'Negative': {'emoji':'😠','color':'#e74c3c'},
}

def clean(text):
    text = re.sub(r'<.*?>',   '', text.lower())
    text = re.sub(r'http\S+','', text)
    text = re.sub(r'[^a-z\s]','',text)
    text = re.sub(r'\s+',' ', text).strip()
    return ' '.join([LEMMATIZER.lemmatize(t) for t in text.split() if t not in STOP_WORDS])

class SimpleBertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        return self.classifier(
            self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        )

@st.cache_resource
def load_ml():
    if not os.path.exists('models/best_classical_model.pkl'): return None, None, None
    with open('models/best_classical_model.pkl','rb') as f: clf        = pickle.load(f)
    with open('models/word_tfidf.pkl',           'rb') as f: word_tfidf = pickle.load(f)
    with open('models/char_tfidf.pkl',           'rb') as f: char_tfidf = pickle.load(f)
    return clf, word_tfidf, char_tfidf

def pred_ml(text, clf, word_tf, char_tf):
    c   = clean(text)
    vec = hstack([word_tf.transform([c]), char_tf.transform([c])])
    idx = int(clf.predict(vec)[0])
    if hasattr(clf,'predict_proba'):
        p = clf.predict_proba(vec)[0]
        conf = {IDX_TO_LABEL[i]:float(p[i])*100 for i in range(NUM_CLASSES)}
    else:
        conf = {IDX_TO_LABEL[i]:(100.0 if i==idx else 0.0) for i in range(NUM_CLASSES)}
    return IDX_TO_LABEL[idx], conf

st.set_page_config(page_title='Feedback Analyzer', page_icon='💬', layout='centered')
st.markdown('<h1 style="text-align:center">💬 Customer Feedback Analyzer</h1>', unsafe_allow_html=True)
st.markdown('---')

text   = st.text_area('Enter Feedback:', placeholder='Type any customer feedback...', height=150)

if st.button('Analyze', use_container_width=True):
    if not text.strip():
        st.warning('Please enter feedback.')
    else:
        with st.spinner('Analyzing...'):
          clf, word_tf, char_tf = load_ml()
          if clf is None: st.error('Classical model not found. Run Section 4.'); st.stop()
          label, conf = pred_ml(text, clf, word_tf, char_tf)
        c = CFG[label]
        st.markdown(
            f'<div style="background:{c["color"]}22;border:2px solid {c["color"]};'
            f'border-radius:12px;padding:1.5rem;text-align:center;margin:1rem 0">'
            f'<div style="font-size:3rem">{c["emoji"]}</div>'
            f'<div style="font-size:2rem;font-weight:700;color:{c["color"]}">{label}</div></div>',
            unsafe_allow_html=True
        )
