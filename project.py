import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

# Initialize the lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase text
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Stem, lemmatize, and remove stopwords
    text = " ".join([stemmer.stem(lemmatizer.lemmatize(word)) for word in text.split() if word not in stop_words])
    return text

def extract_text_from_csv(file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    return df

def compute_weighted_tfidf(resumes, job_descriptions, weights):
  
    vectorizer = TfidfVectorizer(
        stop_words='english', max_df=0.9, min_df=1, 
        ngram_range=(1, 2), max_features=10000, sublinear_tf=True
    )
    # Combine all text for fitting
    all_texts = pd.concat([resumes['text'], job_descriptions['description']])
    vectorizer.fit(all_texts)
    
    resume_vectors = vectorizer.transform(resumes['text'])
    job_desc_vectors = vectorizer.transform(job_descriptions['description'])
    
    # Apply weights to TF-IDF vectors
    weighted_resume_vectors = resume_vectors.multiply(weights['resume'])
    weighted_job_desc_vectors = job_desc_vectors.multiply(weights['job_description'])
    
    return weighted_resume_vectors, weighted_job_desc_vectors

def compute_similarity(resume_vectors, job_desc_vectors):
   
    return cosine_similarity(resume_vectors, job_desc_vectors)

def main():
    st.title("Enhanced Weighted Resume Screening System")

    st.write("Upload your resume and job description files in CSV format.")
    
    resume_file = st.file_uploader("Upload Resumes CSV", type="csv")
    job_desc_file = st.file_uploader("Upload Job Descriptions CSV", type="csv")
    
    if resume_file and job_desc_file:
        # Extract text from CSV files
        resumes_df = extract_text_from_csv(resume_file)
        job_desc_df = extract_text_from_csv(job_desc_file)
        
        
        if 'text' not in resumes_df.columns:
            st.error("The 'text' column is missing in the resumes CSV file.")
            return
        if 'description' not in job_desc_df.columns:
            st.error("The 'description' column is missing in the job descriptions CSV file.")
            return
        
        # Create DataFrames with the appropriate columns
        resumes = pd.DataFrame({'text': resumes_df['text'].apply(preprocess_text)})
        job_descriptions = pd.DataFrame({'description': job_desc_df['description'].apply(preprocess_text)})
        
        # Set custom weights for resume and job description sections (customize as needed)
        weights = {
            'resume': 1.2,  # Apply higher weight to resume vectors
            'job_description': 1.0  # Base weight for job description vectors
        }
        
        # Compute weighted TF-IDF vectors and similarity
        resume_vectors, job_desc_vectors = compute_weighted_tfidf(resumes, job_descriptions, weights)
        similarities = compute_similarity(resume_vectors, job_desc_vectors)
        
        # Prepare results
        results = []
        for i in range(len(resumes)):
            for j in range(len(job_descriptions)):
                results.append({
                    'Resume ID': i + 1,
                    'Job Description ID': j + 1,
                    'Weighted Similarity Score': similarities[i, j]
                })
        
        # Convert results to DataFrame and sort by similarity score in descending order
        results_df = pd.DataFrame(results).sort_values(by='Weighted Similarity Score', ascending=False)
        
        st.write("Weighted Similarity Scores (Sorted by Descending Order):")
        st.dataframe(results_df)
        
        # Save sorted results as a downloadable CSV
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Sorted Weighted Similarity Scores",
            data=csv,
            file_name='sorted_weighted_similarity_scores.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
