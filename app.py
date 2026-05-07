"""
Amazon Review Analyzer — Streamlit Web App
IronHack Project — Domien Darmont

Run with: streamlit run app.py

Prerequisites:
    pip install streamlit pandas numpy plotly transformers torch
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from transformers import pipeline

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title            = "Amazon Review Analyzer",
    page_icon             = "🛍️",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# ── Color palette ────────────────────────────────────────────
C = {
    'positive'    : '#66BB6A',
    'positive_bg' : '#F1F8E9',
    'neutral'     : '#FFB74D',
    'neutral_bg'  : '#FFF3E0',
    'negative'    : '#EF5350',
    'negative_bg' : '#FFEBEE',
    'primary'     : '#1565C0',
    'text'        : '#1A1A1A',
    'muted'       : '#757575',
    'border'      : '#E8E6E1',
    'bg'          : '#F8F7F4',
    'white'       : '#FFFFFF',
}

CATEGORY_ICONS = {
    'Fire Tablets'         : '📱',
    'Batteries'            : '🔋',
    'Echo & Smart Speakers': '🎙️',
    'Kindle E-readers'     : '📚',
    'Fire TV & Accessories': '📺',
}

BADGES = [
    ('Top Pick',     '#E3F2FD', '#1565C0'),
    ('Best Value',   '#F1F8E9', '#2E7D32'),
    ('Premium Pick', '#F3E5F5', '#6A1B9A'),
]

# ── Demo examples ─────────────────────────────────────────────
DEMO_EXAMPLES = {
    'pos1': "This Kindle is absolutely amazing! The screen is crystal clear and the battery lasts for weeks. Best purchase I ever made, highly recommend!",
    'pos2': "The Echo is a game changer! Alexa understands everything perfectly and the sound quality is outstanding. Love it in every room of my house!",
    'neu1': "The Fire Tablet does what it promises but nothing more. Performance is average and the battery life is acceptable. Okay for the price I guess.",
    'neu2': "These batteries are decent. Not as long lasting as Duracell but cheaper. They work fine for remote controls and basic devices.",
    'neg1': "Very disappointed with these batteries. They died within a week in my remote control. Expected much better quality from Amazon. Will not buy again.",
    'neg2': "The Fire TV is a complete waste of money. The interface is confusing, apps crash constantly and Alexa never understands my commands. Returning it tomorrow.",
}

# ── Custom CSS ───────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
    h1, h2, h3 {{ font-family: 'DM Serif Display', serif; color: {C['text']}; }}
    .stApp {{ background-color: {C['bg']}; }}
    .metric-card {{
        background: {C['white']};
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        border: 1px solid {C['border']};
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 8px;
    }}
    .metric-label {{
        font-size: 11px;
        color: {C['muted']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 600;
        color: {C['text']};
        line-height: 1.1;
    }}
    .product-card {{
        background: {C['white']};
        border-radius: 16px;
        border: 1px solid {C['border']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        overflow: hidden;
        height: 100%;
    }}
    .product-icon-box {{
        background: #FAFAFA;
        border-bottom: 1px solid {C['border']};
        height: 160px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 72px;
    }}
    .product-body {{ padding: 1.2rem 1.4rem 1.4rem; }}
    .product-badge {{
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 10px;
    }}
    .product-name {{
        font-family: 'DM Serif Display', serif;
        font-size: 17px;
        font-weight: 700;
        color: {C['text']};
        margin: 0 0 6px;
        line-height: 1.4;
        text-decoration: underline;
        text-decoration-color: #CCC;
    }}
    .product-rating {{ font-size: 13px; color: {C['muted']}; margin-bottom: 12px; }}
    .product-summary {{
        font-size: 13px;
        color: #333;
        line-height: 1.6;
        border-top: 1px solid {C['border']};
        padding-top: 10px;
    }}
    .article-box {{
        background: {C['white']};
        border-radius: 14px;
        padding: 2.5rem;
        border: 1px solid {C['border']};
        line-height: 1.9;
        color: {C['text']};
        font-size: 15px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    .counter-box {{
        border-radius: 14px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1rem;
        border: 2px solid;
    }}
    .demo-box {{
        background: {C['white']};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid {C['border']};
        margin-bottom: 12px;
    }}
    div[data-testid="stSidebarNav"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'processed')

@st.cache_data
def load_data():
    full_clustered = os.path.join(DATA_DIR, 'reviews_full_clustered.csv')
    clustered_path = os.path.join(DATA_DIR, 'clustered_reviews.csv')
    clustered  = pd.read_csv(full_clustered if os.path.exists(full_clustered) else clustered_path)
    classified = pd.read_csv(os.path.join(DATA_DIR, 'classified_reviews.csv'))
    articles   = pd.read_csv(os.path.join(DATA_DIR, 'articles.csv'))
    rouge      = pd.read_csv(os.path.join(DATA_DIR, 'rouge_scores.csv'))
    for df in [clustered, classified]:
        if 'product_name' in df.columns:
            df['product_name'] = (
                df['product_name']
                .str.split('\n').str[0]
                .str.replace(r',+$',  '', regex=True)
                .str.replace(r',,,+', '', regex=True)
                .str.strip()
            )
    return clustered, classified, articles, rouge

clustered, classified, articles_df, rouge_df = load_data()
articles_dict = dict(zip(articles_df['category'], articles_df['article']))
categories    = sorted([c for c in clustered['cluster_label'].dropna().unique() if c != 'Other'])
TOTAL_BASE    = len(clustered)

# ── Model ────────────────────────────────────────────────────
FINETUNED = os.path.join(os.path.dirname(__file__), 'data', 'finetuned_model', 'bert_sentiment_finetuned')

@st.cache_resource
def load_classifier():
    path = FINETUNED if os.path.exists(FINETUNED) else 'nlptown/bert-base-multilingual-uncased-sentiment'
    return pipeline('text-classification', model=path, tokenizer=path,
                    truncation=True, max_length=128, device=-1)

def map_label(label: str) -> str:
    if label in ['positive', 'neutral', 'negative']:
        return label
    if 'LABEL' in label:
        return {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}.get(label, 'neutral')
    try:
        stars = int(label.split()[0])
        return 'negative' if stars <= 2 else ('neutral' if stars == 3 else 'positive')
    except:
        return 'neutral'

def get_top3(df_cat: pd.DataFrame) -> pd.DataFrame:
    return (
        df_cat.groupby('product_name')
        .agg(avg_rating=('rating', 'mean'), review_count=('rating', 'count'))
        .reset_index()
        .assign(score=lambda x: x['avg_rating'] * np.log1p(x['review_count']))
        .sort_values('score', ascending=False)
        .head(3)
        .reset_index(drop=True)
    )

def get_short_summary(article: str, product_name: str) -> str:
    for line in article.split('\n'):
        l = line.strip()
        if len(l) > 40 and product_name[:20].lower() in l.lower():
            return l[:160] + '...' if len(l) > 160 else l
    for line in article.split('\n'):
        l = line.strip()
        if len(l) > 60 and not l.startswith('#') and not l.startswith('**'):
            return l[:160] + '...' if len(l) > 160 else l
    return ''

def colorize_article(text: str) -> str:
    lines, colored = text.split('\n'), []
    for line in lines:
        s, l = line.strip(), line.strip().lower()
        if l.startswith('pros:') or '**pros' in l:
            s = f'<span style="color:#388E3C;font-weight:500">✅ {s}</span>'
        elif l.startswith('cons:') or '**cons' in l:
            s = f'<span style="color:#D32F2F;font-weight:500">⚠️ {s}</span>'
        elif l.startswith('who should buy') or '**who should' in l:
            s = f'<span style="color:#1565C0;font-weight:500">👤 {s}</span>'
        else:
            s = f'<span style="color:{C["text"]}">{s}</span>'
        colored.append(s)
    return '<br>'.join(colored)

# ── Session state init ────────────────────────────────────────
if 'demo_text' not in st.session_state:
    st.session_state.demo_text = ''
if 'new_reviews' not in st.session_state:
    st.session_state.new_reviews = []

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Review Analyzer")
    st.markdown("*IronHack Project Week 6*")
    st.markdown("Domien Darmont")
    st.markdown("Analyze and visualize Amazon product reviews with real-time sentiment classification.")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Dashboard",
        "🏆 Top Products",
        "📂 Classify Review",
        "📝 Add Review"
    ], label_visibility="collapsed")
    st.markdown("---")
    total = TOTAL_BASE + len(st.session_state.get('new_reviews', []))
    st.markdown(f"**Reviews:** {total:,}")
    st.markdown(f"**Categories:** {len(categories)}")
    st.markdown(f"**Models:** BERT · Llama3.2")
    st.markdown("---")
    st.markdown("<small style='color:#999'>Sentiment: Fine-tuned BERT<br>Clustering: KMeans + all-mpnet<br>Summarization: Llama3.2 3B</small>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("Product Review Dashboard")
    st.markdown("Real-time insights into customer sentiment across Amazon product categories.")
    st.markdown("---")

    total    = TOTAL_BASE + len(st.session_state.get('new_reviews', []))
    s_col    = 'ground_truth' if 'ground_truth' in clustered.columns else 'bert_sentiment'
    pos_pct  = (clustered[s_col] == 'positive').mean() * 100 if s_col in clustered.columns else 0
    best_cat = clustered[clustered['cluster_label'] != 'Other'].groupby('cluster_label')['rating'].mean().idxmax()
    avg_rouge= rouge_df['rouge1'].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, color in [
        (c1, "Total Reviews",       f"{total:,}",       C['text']),
        (c2, "Positive Reviews",    f"{pos_pct:.1f}%",  C['positive']),
        (c3, "Best Rated Category", best_cat,           C['primary']),
        (c4, "Avg ROUGE-1",         f"{avg_rouge:.3f}", C['primary']),
    ]:
        with col:
            fs = '18px' if len(str(value)) > 12 else '28px'
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color};font-size:{fs}">{value}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    selected_cat = st.selectbox("Select a product category", categories)
    df_cat = clustered[clustered['cluster_label'] == selected_cat]

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Rating Distribution")
        rc  = df_cat['rating'].value_counts().sort_index()
        fig = px.bar(x=rc.index, y=rc.values, color=rc.index,
                     color_discrete_sequence=['#EF9A9A','#FFCC80','#FFF176','#A5D6A7','#66BB6A'],
                     labels={'x':'Stars','y':'Reviews'})
        fig.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white',
                          font_family='DM Sans', margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Sentiment Distribution")
        if s_col in df_cat.columns:
            sent = df_cat[s_col].value_counts()
            fig2 = px.pie(values=sent.values, names=sent.index, color=sent.index,
                          color_discrete_map={'positive':C['positive'],'neutral':C['neutral'],'negative':C['negative']},
                          hole=0.48)
            fig2.update_layout(font_family='DM Sans', margin=dict(t=10,b=10), paper_bgcolor='white')
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top Product per Category")
    top_per_cat = []
    for cat in categories:
        df_c = clustered[clustered['cluster_label'] == cat]
        if len(df_c) > 0 and 'product_name' in df_c.columns:
            top = (
                df_c.groupby('product_name')
                .agg(avg_rating=('rating','mean'), review_count=('rating','count'))
                .reset_index()
                .assign(score=lambda x: x['avg_rating'] * np.log1p(x['review_count']))
                .sort_values('score', ascending=False)
                .iloc[0]
            )
            top_per_cat.append({
                'Category'    : cat,
                'Top Product' : top['product_name'][:55],
                'Rating'      : f"{top['avg_rating']:.1f} ⭐",
                'Reviews'     : f"{int(top['review_count']):,}",
            })

    top_df = pd.DataFrame(top_per_cat)
    st.dataframe(top_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — TOP PRODUCTS
# ══════════════════════════════════════════════════════════════
elif page == "🏆 Top Products":
    st.title("Top Products")
    st.markdown("Our top product recommendations per category, based on customer reviews.")
    st.markdown("---")

    selected_cat = st.selectbox("Select a product category", categories)
    df_cat  = clustered[clustered['cluster_label'] == selected_cat]
    top3    = get_top3(df_cat)
    icon    = CATEGORY_ICONS.get(selected_cat, '🛍️')
    article = articles_dict.get(selected_cat, '')

    cols = st.columns(3)
    for i, (col, (_, row)) in enumerate(zip(cols, top3.iterrows())):
        badge_label, badge_bg, badge_color = BADGES[i]
        summary = get_short_summary(article, row['product_name'])
        stars   = '⭐' * round(row['avg_rating'])
        with col:
            st.markdown(f"""
            <div class="product-card">
                <div class="product-icon-box">{icon}</div>
                <div class="product-body">
                    <span class="product-badge" style="background:{badge_bg};color:{badge_color}">{badge_label}</span>
                    <div class="product-name">{row['product_name'][:55]}</div>
                    <div class="product-rating">{stars} {row['avg_rating']:.1f}/5 · {row['review_count']:,} reviews</div>
                    <div class="product-summary">{summary}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    if article:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Full Analysis")
        rouge_row = rouge_df[rouge_df['category'] == selected_cat]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Reviews analyzed", f"{len(df_cat):,}")
        with c2:
            avg_r = df_cat['rating'].mean() if 'rating' in df_cat.columns else 0
            st.metric("Average rating", f"{avg_r:.1f} ⭐")
        with c3:
            if not rouge_row.empty:
                st.metric("ROUGE-1", f"{rouge_row['rouge1'].values[0]:.3f}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="article-box">{colorize_article(article)}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 3 — CLASSIFY REVIEW
# ══════════════════════════════════════════════════════════════
elif page == "📂 Classify Review":
    st.title("Live Review Classifier")
    st.markdown("Type a review or pick a demo example — our fine-tuned BERT model classifies its sentiment in real time.")
    st.markdown("---")

    # ── Demo examples ────────────────────────────────────────
    st.markdown("**Try a demo example:**")
    col_pos, col_neu, col_neg = st.columns(3)

    with col_pos:
        st.markdown(f"<div style='color:{C['positive']};font-weight:600;margin-bottom:6px'>✅ Positive</div>", unsafe_allow_html=True)
        if st.button("Kindle review", key="pos1"):
            st.session_state.demo_text = DEMO_EXAMPLES['pos1']
        if st.button("Echo review", key="pos2"):
            st.session_state.demo_text = DEMO_EXAMPLES['pos2']

    with col_neu:
        st.markdown(f"<div style='color:{C['neutral']};font-weight:600;margin-bottom:6px'>➖ Neutral</div>", unsafe_allow_html=True)
        if st.button("Fire Tablet review", key="neu1"):
            st.session_state.demo_text = DEMO_EXAMPLES['neu1']
        if st.button("Batteries review", key="neu2"):
            st.session_state.demo_text = DEMO_EXAMPLES['neu2']

    with col_neg:
        st.markdown(f"<div style='color:{C['negative']};font-weight:600;margin-bottom:6px'>❌ Negative</div>", unsafe_allow_html=True)
        if st.button("Batteries negative", key="neg1"):
            st.session_state.demo_text = DEMO_EXAMPLES['neg1']
        if st.button("Fire TV negative", key="neg2"):
            st.session_state.demo_text = DEMO_EXAMPLES['neg2']

    st.markdown("<br>", unsafe_allow_html=True)

    review_input = st.text_area(
        "Or type your own review",
        value=st.session_state.demo_text,
        placeholder="e.g. This Kindle is absolutely amazing, best purchase I ever made!",
        height=130
    )

    if st.button("Analyze Review", type="primary"):
        if review_input.strip():
            with st.spinner("Analyzing with fine-tuned BERT..."):
                try:
                    clf        = load_classifier()
                    result     = clf(review_input[:512])[0]
                    sentiment  = map_label(result['label'])
                    confidence = result['score']

                    category = categories[0]
                    for cat in categories:
                        if any(w in review_input.lower() for w in cat.lower().split()):
                            category = cat
                            break

                    s_color = {'positive':C['positive'],'neutral':C['neutral'],'negative':C['negative']}
                    s_emoji = {'positive':'✅','neutral':'➖','negative':'❌'}
                    s_bg    = {'positive':C['positive_bg'],'neutral':C['neutral_bg'],'negative':C['negative_bg']}

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""<div class="metric-card" style="background:{s_bg[sentiment]}">
                            <div class="metric-label">Sentiment</div>
                            <div class="metric-value" style="color:{s_color[sentiment]}">{s_emoji[sentiment]} {sentiment.capitalize()}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value">{confidence:.1%}</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""<div class="metric-card">
                            <div class="metric-label">Category</div>
                            <div class="metric-value" style="font-size:17px">{category}</div>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.progress(confidence)

                except Exception as e:
                    st.error(f"Model error: {e}")
        else:
            st.warning("Please enter a review or pick a demo example first.")

    st.markdown("---")
    st.subheader("Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Accuracy**")
        st.markdown("""
| Model | Accuracy |
|-------|----------|
| Out-of-the-Box BERT | 77.0% |
| Fine-tuned BERT | **79.4%** |
        """)
    with c2:
        st.markdown("**F1-Score per Class**")
        st.markdown("""
| Class | F1 |
|-------|-----|
| Positive | 0.869 |
| Neutral | 0.689 |
| Negative | 0.817 |
        """)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — ADD REVIEW
# ══════════════════════════════════════════════════════════════
elif page == "📝 Add Review":
    st.title("Add Your Review")
    st.markdown("Submit a product review — it will be automatically classified by BERT and added live.")
    st.markdown("---")

    col_form, col_recent = st.columns(2)

    with col_form:
        st.subheader("Submit a Review")
        product_name      = st.text_input("Product name", placeholder="e.g. Amazon Kindle Paperwhite")
        selected_category = st.selectbox("Product category", categories)
        rating            = st.slider("Rating", 1, 5, 4)
        review_text       = st.text_area("Your review", placeholder="Write your review here...", height=130)

        if st.button("Submit Review", type="primary"):
            if review_text.strip() and product_name.strip():
                with st.spinner("Classifying with BERT..."):
                    try:
                        clf        = load_classifier()
                        result     = clf(review_text[:512])[0]
                        sentiment  = map_label(result['label'])
                        confidence = result['score']

                        st.session_state.new_reviews.insert(0, {
                            'product_name' : product_name,
                            'category'     : selected_category,
                            'rating'       : rating,
                            'review_text'  : review_text,
                            'sentiment'    : sentiment,
                            'confidence'   : confidence,
                        })

                        new_total = TOTAL_BASE + len(st.session_state.new_reviews)
                        color = {'positive':C['positive'],'neutral':C['neutral'],'negative':C['negative']}[sentiment]
                        bg    = {'positive':C['positive_bg'],'neutral':C['neutral_bg'],'negative':C['negative_bg']}[sentiment]
                        bezos = {
                            'positive': ("💰 Jeff Bezos just gained $0.000001 in market value.", f"Review #{new_total} added. The algorithm approves. ✅"),
                            'neutral':  ("🤷 Jeff Bezos remains unbothered.", "3 stars. The most mysterious rating of them all. 🔮"),
                            'negative': ("📉 Jeff Bezos just lost $0.000001 in market value.", "Noted. Someone is not happy. 😬"),
                        }
                        line1, line2 = bezos[sentiment]

                        st.markdown(f"""
                        <div class="counter-box" style="background:{bg};border-color:{color}">
                            <div style="font-size:20px;font-weight:600;color:{color};margin-bottom:6px">{line1}</div>
                            <div style="font-size:14px;color:{C['text']};margin-bottom:16px">{line2}</div>
                            <div style="font-size:42px;font-weight:700;color:{color};font-family:'DM Serif Display',serif">{new_total:,}</div>
                            <div style="font-size:11px;color:{C['muted']};text-transform:uppercase;letter-spacing:0.08em;margin-top:4px">
                                Total Reviews · +{len(st.session_state.new_reviews)} this session
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please fill in both the product name and review.")

    with col_recent:
        st.subheader(f"Recently Added ({len(st.session_state.new_reviews)})")
        if st.session_state.new_reviews:
            for review in st.session_state.new_reviews[:6]:
                s      = review['sentiment']
                border = {'positive':C['positive'],'neutral':C['neutral'],'negative':C['negative']}.get(s,'#999')
                bg     = {'positive':C['positive_bg'],'neutral':C['neutral_bg'],'negative':C['negative_bg']}.get(s,'#FFF')
                emoji  = {'positive':'✅','neutral':'➖','negative':'❌'}.get(s,'⚪')
                stars  = '⭐' * review['rating']
                st.markdown(f"""
                <div style="background:{bg};border-radius:12px;padding:1rem 1.2rem;
                    border-left:4px solid {border};margin-bottom:10px;color:{C['text']}">
                    <b style="font-size:14px">{review['product_name']}</b> · {stars}<br>
                    <span style="color:{C['muted']};font-size:12px">{review['category']}</span><br>
                    <span style="font-size:13px">{review['review_text'][:120]}{'...' if len(review['review_text'])>120 else ''}</span><br>
                    <span style="font-size:12px;color:{C['muted']}">{emoji} {s.capitalize()} · {review['confidence']:.1%} confidence</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Live Sentiment Distribution")
            sent_s = pd.Series([r['sentiment'] for r in st.session_state.new_reviews]).value_counts()
            fig = px.pie(values=sent_s.values, names=sent_s.index, color=sent_s.index,
                         color_discrete_map={'positive':C['positive'],'neutral':C['neutral'],'negative':C['negative']},
                         hole=0.48)
            fig.update_layout(margin=dict(t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', font_family='DM Sans')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No reviews added yet. Submit your first review on the left!")
