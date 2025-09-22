import pandas as pd
import gradio as gr
import gensim
from gensim import corpora
import spacy  # For advanced preprocessing
from spacy.cli import download
import nltk
from nltk.corpus import stopwords  # For custom stopword integration
import time  # For timing and debugging

# Try to load, if not available then download dynamically
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

nltk.download('stopwords')  # Ensure stopwords are downloaded

# Function to load data from uploaded file (supports CSV, JSON, JSONL)
def load_data(file_obj):
    if file_obj is None:
        return None
    ext = file_obj.name.split('.')[-1].lower()
    if ext == 'jsonl':
        return pd.read_json(file_obj, lines=True)
    elif ext == 'json':
        return pd.read_json(file_obj)
    elif ext == 'csv':
        return pd.read_csv(file_obj)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Please upload CSV, JSON, or JSONL files.")

# Function to clean and filter data (adapted for review-only focus)
def clean_and_filter_data(df):
    # Standardize column names
    df.columns = df.columns.str.lower()
    df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns

    # Map columns to required names
    standardized_cols = {}
    required_substrings = {
        'rating': 'rating',
        'text': ['text', 'review'],
        'verified_purchase': ['verified', 'purchase'],
        'helpful_vote': ['helpful', 'vote'],
    }

    for col in df.columns:
        for key, substr in required_substrings.items():
            if isinstance(substr, list):
                if all(s in col for s in substr):
                    standardized_cols[key] = col
                    break
            elif substr in col:
                standardized_cols[key] = col
                break

    df = df.rename(columns={v: k for k, v in standardized_cols.items()})

    # Keep only required columns
    cols_to_keep = ['text', 'rating', 'verified_purchase', 'helpful_vote']
    df = df[[col for col in cols_to_keep if col in df.columns]]

    # Filter verified purchases if column exists
    if 'verified_purchase' in df.columns:
        df = df[df['verified_purchase'] == True]

    # Add word count column
    if 'text' in df.columns:
        df['word_count'] = df['text'].astype(str).str.split().str.len()
        df = df[df['word_count'] >= 5]

    # Remove duplicates
    df = df.drop_duplicates()

    return df

# Optimized preprocess_text
def preprocess_text(text):
    if not text or not text.strip():
        return []
    stop_words = set(stopwords.words('english'))
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words and len(token) > 2]
    return tokens

# Optimized perform_lda
def perform_lda(df, num_topics=3, passes=1, num_words=10, max_reviews=1000):
    if df.empty or 'text' not in df.columns:
        return "No text data available for LDA."
    texts = df['text'].dropna().tolist()
    # Sample only first max_reviews reviews for speed
    texts = texts[:max_reviews]
    processed_texts = [preprocess_text(text) for text in texts if text.strip()]
    if not processed_texts or len(processed_texts) < 10:
        return "Too few valid texts after preprocessing (need 10+ reviews)."
    dictionary = corpora.Dictionary(processed_texts)
    if len(dictionary) < 20:
        return "Vocabulary too small for reliable LDA."
    dictionary.filter_extremes(no_below=1, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        alpha='auto',
        per_word_topics=True,
        chunksize=100  # Faster training
    )
    topics = lda_model.print_topics(num_words=num_words)
    topic_summary = f"LDA completed with {num_topics} topics on {len(processed_texts)} reviews.\n\n"
    for topic_id, topic in enumerate(topics):
        words = ', '.join([word.split('*')[1].replace('"', '') for word in topic[1].split('+')])
        topic_summary += f"Topic {topic_id}: {words}\n"
    return topic_summary

# Function to clean and prepare data (with timing)
def clean_data(file_obj, max_rows_display=10):
    start_time = time.time()
    try:
        df = load_data(file_obj)
        if df is None:
            return "Error: No file uploaded.", pd.DataFrame()
        load_time = time.time() - start_time
        
        cleaned_df = clean_and_filter_data(df)
        clean_time = time.time() - load_time - start_time
        
        if cleaned_df.empty:
            total_time = time.time() - start_time
            return f"No data after cleaning.\nLoad: {load_time:.2f}s | Clean: {clean_time:.2f}s | Total: {total_time:.2f}s", pd.DataFrame()
        
        total_time = time.time() - start_time
        summary = f"✅ Cleaning done in {total_time:.2f}s!\n• Original reviews: {len(df)}\n• After cleaning: {len(cleaned_df)}\n• Load time: {load_time:.2f}s | Clean time: {clean_time:.2f}s"
        display_df = cleaned_df.head(max_rows_display).fillna("")
        
        return summary, display_df, cleaned_df  # Return cleaned_df as state
    except Exception as e:
        total_time = time.time() - start_time
        return f"❌ Error in cleaning: {str(e)}\nTotal time: {total_time:.2f}s", pd.DataFrame(), pd.DataFrame()

# Function to run LDA (with timing, uses state)
def run_lda(cleaned_df, num_topics=5, passes=2, num_words=10):
    if cleaned_df is None or cleaned_df.empty:
        return "No cleaned data from previous step."
    start_time = time.time()
    try:
        lda_results = perform_lda(cleaned_df, num_topics, passes, num_words)
        lda_time = time.time() - start_time
        instruction = (
            "\n\n---\n"
            "**Tip:** For a more human-like interpretation, copy the topics below and paste them into ChatGPT or another AI assistant. Describe your product if necessary.\n "
            "Prompt with:\n 'Interpret these LDA topics in plain English. What themes do they represent?'\n"
        )
        summary = f"✅ LDA done in {lda_time:.2f}s! Results below."
        return summary, lda_results
    except Exception as e:
        lda_time = time.time() - start_time
        return f"❌ Error in LDA: {str(e)}\nTime: {lda_time:.2f}s", ""

# Gradio Demo with Separate Buttons for Step-by-Step Processing
with gr.Blocks() as demo:
    gr.Markdown("# Step-by-Step Review LDA Analyzer (with Timing)")
    gr.Markdown(
        "Upload your reviews file, then click buttons sequentially to isolate issues: Clean first, then Run LDA. "
        "Timing is shown for each step to pinpoint slowdowns."
    )
    
    file_input = gr.File(label="Upload Reviews File", file_count="single")
    max_rows_input = gr.Number(value=10, label="Max Rows to Show in Preview", minimum=1, maximum=100)
    
    with gr.Row():
        num_topics = gr.Number(value=3, label="Number of Topics", minimum=1, maximum=20)
        passes = gr.Number(value=1, label="Passes (Iterations)", minimum=1, maximum=10)
        num_words = gr.Number(value=10, label="Words per Topic", minimum=5, maximum=20)
    
    clean_button = gr.Button("Step 1: Clean and Filter Data")
    summary_clean_output = gr.Textbox(label="Step 1 Summary & Timing", lines=5, interactive=False)
    dataframe_output = gr.Dataframe(label="Step 1: Cleaned Data Preview", interactive=False)
    
    lda_button = gr.Button("Step 2: Run LDA")
    summary_lda_output = gr.Textbox(label="Step 2 Summary & Timing", lines=2, interactive=False)
    lda_output = gr.Textbox(label="Step 2: LDA Topics", lines=20, interactive=False)
    
    # State to hold cleaned DataFrame between steps
    cleaned_state = gr.State()
    
    clean_button.click(
        fn=clean_data,
        inputs=[file_input, max_rows_input],
        outputs=[summary_clean_output, dataframe_output, cleaned_state]
    )
    
    lda_button.click(
        fn=run_lda,
        inputs=[cleaned_state, num_topics, passes, num_words],
        outputs=[summary_lda_output, lda_output]
    )

if __name__ == "__main__":
    demo.launch()