"""
AI Sales Copy & Ad Content Agent - Ultra Lightweight
====================================================
No heavy dependencies - Fast loading guaranteed
"""

import streamlit as st
import os
import json
import sqlite3
import re
from datetime import datetime

# =============================================================================
# PAGE CONFIG - MUST BE FIRST
# =============================================================================
st.set_page_config(
    page_title="AI Sales Copy Agent",
    page_icon="🚀",
    layout="wide"
)

# =============================================================================
# DATABASE
# =============================================================================
def get_db():
    conn = sqlite3.connect('sales.db', check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY, business TEXT, type TEXT, product TEXT, 
        audience TEXT, offer TEXT, tone TEXT, response TEXT, created TEXT
    )''')
    return conn

def save_history(conn, inputs, response):
    conn.execute('INSERT INTO history (business, type, product, audience, offer, tone, response, created) VALUES (?,?,?,?,?,?,?,?)',
        (inputs['business_name'], inputs['business_type'], inputs['product_service'], 
         inputs['target_audience'], inputs['offer'], inputs['tone'], json.dumps(response), datetime.now().isoformat()))
    conn.commit()

def get_history(conn):
    return conn.execute('SELECT * FROM history ORDER BY id DESC LIMIT 20').fetchall()

# =============================================================================
# KEYWORD EXTRACTION (Simple - No NLTK)
# =============================================================================
STOPWORDS = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','is','are','was','were','be','been','have','has','had','do','does','did','will','would','could','should','may','might','must','can','this','that','these','those','i','you','he','she','it','we','they','who','what','which','when','where','why','how','all','each','every','both','few','more','most','other','some','such','no','not','only','own','same','so','than','too','very','just','your','our','their','my','his','her','its','as','if'}

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    freq = {}
    for w in words:
        if w not in STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:12]]

def make_hashtags(keywords):
    return [f"#{k}" for k in keywords[:8]] + ['#marketing', '#business']

# =============================================================================
# PROMPT
# =============================================================================
TONES = {
    'Professional': "formal, authoritative",
    'Emotional': "storytelling, feelings",
    'Exciting': "energetic, dynamic",
    'Urgent': "scarcity, time-sensitive",
    'Friendly': "warm, conversational",
    'Luxury': "sophisticated, premium"
}

def make_prompt(i):
    return f"""Generate marketing content as JSON only:

Business: {i['business_name']} ({i['business_type']})
Product: {i['product_service']}
Audience: {i['target_audience']}
Offer: {i['offer']}
Tone: {i['tone']} ({TONES.get(i['tone'],'')})

Return this exact JSON structure:
{{"google_ads":{{"headlines":["5 headlines max 30 chars"],"descriptions":["3 desc max 90 chars"]}},"facebook":{{"text":["2 ad texts"],"headlines":["3 headlines"]}},"instagram":{{"captions":["2 captions with emojis"],"hashtags":["15 hashtags"]}},"seo":{{"titles":["3 titles 50-60 chars"],"meta":["3 descriptions 150-160 chars"]}},"landing":{{"headline":"main","subheadline":"supporting","ctas":["3 buttons"],"benefits":["4 points"]}},"keywords":["10 keywords"],"ctas":["5 call-to-actions"]}}

ONLY JSON. No markdown. No explanation."""

# =============================================================================
# GENERATE
# =============================================================================
def generate(api_key, inputs):
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":make_prompt(inputs)}],
            max_tokens=2500
        )
        content = r.choices[0].message.content.strip()
        if content.startswith('```'):
            content = re.sub(r'^```\w*\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        return json.loads(content)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# =============================================================================
# UI
# =============================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.title("🚀 AI Sales Agent")
        page = st.radio("", ["🏠 Home", "✨ Generate", "📊 History"])
        st.markdown("---")
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get('key',''))
        if api_key:
            st.session_state['key'] = api_key
            st.success("✅ Key set")

    # Home
    if page == "🏠 Home":
        st.title("🚀 AI Sales Copy & Ad Content Agent")
        st.markdown("### Generate Marketing Content for All Platforms")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**🎯 Google Ads**\nHeadlines & Descriptions")
        c2.markdown("**📱 Social Media**\nFacebook & Instagram")
        c3.markdown("**🔍 SEO**\nTitles & Meta Tags")
        st.info("👈 Enter API key and go to Generate!")

    # Generate
    elif page == "✨ Generate":
        st.title("✨ Generate Content")
        
        if not st.session_state.get('key'):
            st.warning("⚠️ Enter OpenAI API key in sidebar!")
            return
        
        c1, c2 = st.columns(2)
        with c1:
            business_name = st.text_input("Business Name *")
            business_type = st.selectbox("Type *", ["E-commerce","SaaS","Service","Consulting","Healthcare","Education","Real Estate","Finance","Restaurant","Fitness","Other"])
            product_service = st.text_area("Product/Service *", height=100)
        with c2:
            target_audience = st.text_area("Target Audience *", height=80)
            offer = st.text_input("Current Offer")
            tone = st.selectbox("Tone", list(TONES.keys()))

        valid = all([business_name, product_service, target_audience])
        
        if st.button("🚀 Generate", type="primary", disabled=not valid, use_container_width=True):
            inputs = {'business_name':business_name, 'business_type':business_type, 'product_service':product_service, 'target_audience':target_audience, 'offer':offer, 'tone':tone}
            
            with st.spinner("Generating..."):
                result = generate(st.session_state['key'], inputs)
            
            if result:
                st.session_state['result'] = result
                st.session_state['inputs'] = inputs
                conn = get_db()
                save_history(conn, inputs, result)
                kw = extract_keywords(f"{product_service} {target_audience}")
                st.session_state['kw'] = kw

        # Show results
        if st.session_state.get('result'):
            st.success("✅ Done!")
            
            # Keywords
            if st.session_state.get('kw'):
                with st.expander("🔑 Keywords", expanded=True):
                    st.write(", ".join(st.session_state['kw']))
                    st.write(" ".join(make_hashtags(st.session_state['kw'])))
            
            # Content sections
            for section, data in st.session_state['result'].items():
                with st.expander(f"📌 {section.upper()}", expanded=True):
                    if isinstance(data, dict):
                        for k, v in data.items():
                            st.markdown(f"**{k}:**")
                            if isinstance(v, list):
                                for i, item in enumerate(v, 1):
                                    st.write(f"{i}. {item}")
                            else:
                                st.write(v)
                    elif isinstance(data, list):
                        for i, item in enumerate(data, 1):
                            st.write(f"{i}. {item}")
                    else:
                        st.write(data)
            
            # Export
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📋 Download JSON", json.dumps(st.session_state['result'], indent=2), "content.json", "application/json")
            with c2:
                # Text export
                txt = f"=== AI Generated Marketing Content ===\n\n"
                txt += f"Business: {st.session_state['inputs']['business_name']}\n"
                txt += f"Type: {st.session_state['inputs']['business_type']}\n\n"
                for s, d in st.session_state['result'].items():
                    txt += f"\n=== {s.upper()} ===\n"
                    txt += json.dumps(d, indent=2) + "\n"
                st.download_button("📄 Download TXT", txt, "content.txt", "text/plain")

    # History
    elif page == "📊 History":
        st.title("📊 History")
        conn = get_db()
        history = get_history(conn)
        if not history:
            st.info("No history yet!")
        else:
            st.metric("Total", len(history))
            for row in history:
                with st.expander(f"📌 {row[1]} - {row[8][:10] if row[8] else ''}"):
                    st.write(f"**Type:** {row[2]} | **Tone:** {row[6]}")
                    try:
                        st.json(json.loads(row[7]))
                    except:
                        st.text(row[7][:300])

if __name__ == "__main__":
    main()
