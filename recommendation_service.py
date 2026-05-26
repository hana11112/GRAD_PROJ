from fastapi import FastAPI, Body
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()


def get_recommendations(user_data, all_products):

    df = pd.DataFrame(all_products).copy()

    if df.empty:
        return []

    
    required_columns = ['min_age', 'max_age', 'category_name', 'target_gender']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0 if col in ['min_age', 'max_age'] else ""

    
    df['min_age'] = pd.to_numeric(df['min_age'], errors='coerce').fillna(0)
    df['max_age'] = pd.to_numeric(df['max_age'], errors='coerce').fillna(100)

    
    try:
        user_age = int(user_data.get('age', 0))
    except:
        return []

   
    df = df[
        (user_age >= df['min_age']) &
        (user_age <= df['max_age'])
    ].copy()

    if df.empty:
        return []

    
    df['category_name'] = df['category_name'].fillna('').str.lower()

    df['target_gender'] = (
        df['target_gender']
        .fillna('')
        .str.lower()
        .replace("unisex", "male female")
    )

    
    user_gender = str(user_data.get('gender', '')).lower()

    
    df['metadata'] = (
        df['target_gender'] + " " +
        df['target_gender'] + " " +
        df['category_name']
    )

    
    user_profile = f"{user_gender} {user_gender}"

    
    try:
        
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df['metadata'])
        user_vector = tfidf.transform([user_profile])

       
        similarity_scores = cosine_similarity(
            user_vector,
            tfidf_matrix
        ).flatten()

        df['score'] = similarity_scores

    except:
        
        df['score'] = 0

    
    recommendations = (
        df.sort_values(by='score', ascending=False)
        .drop_duplicates(subset="category_name")
    )

    
    return (
        recommendations
        .head(5)
        .drop(columns=['metadata'])
        .to_dict(orient='records')
    )



@app.post("/recommend")
async def recommend_endpoint(data: dict = Body(...)):
    try:
        user_data = data.get("user_data", {})
        products = data.get("products", [])

        recommendations = get_recommendations(user_data, products)

        return {
            "products": recommendations
        }

    except Exception as e:
        return {
            "error": str(e),
            "products": []
        }