"""
AI Sales Copy & Ad Content Agent - Lightweight Version
======================================================
Optimized for fast loading on Streamlit Cloud
"""

import streamlit as st
import os
import json
import sqlite3
import re
import io
from datetime import datetime

# =============================================================================
# PAGE CONFIG - MUST BE FIRST
# =============================================================================
st.set_page_config(
    page_title="AI Sales Copy Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LAZY IMPORTS - Load only when needed
# =============================================================================
@st.cache_resource
def load_openai():
    """Lazy load OpenAI"""
    import openai
    return openai

@st.cache_resource
def load_docx():
    """Lazy load python-docx"""
    from docx import Document
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    return Document, Pt, WD_ALIGN_PARAGRAPH

@st.cache_resource
def load_reportlab():
    """Lazy load reportlab"""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    return SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, getSampleStyleSheet, ParagraphStyle, colors, inch, A4

# =============================================================================
# DATABASE SETUP
# =============================================================================
@st.cache_resource
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('sales_content.db', check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS content_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            business_name TEXT,
            business_type TEXT,
            product_service TEXT,
            target_audience TEXT,
            offer TEXT,
            tone TEXT,
            platform TEXT,
            full_response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def save_to_history(conn, inputs, outputs):
    """Save generated content to database"""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO content_history 
        (business_name, business_type, product_service, target_audience, offer, tone, platform, full_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        inputs.get('business_name', ''),
        inputs.get('business_type', ''),
        inputs.get('product_service', ''),
        inputs.get('target_audience', ''),
        inputs.get('offer', ''),
        inputs.get('tone', ''),
        str(inputs.get('platform', '')),
        json.dumps(outputs)
    ))
    conn.commit()

def get_history(conn, limit=20):
    """Get content history"""
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM content_history ORDER BY created_at DESC LIMIT ?', (limit,))
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

# =============================================================================
# SIMPLE KEYWORD EXTRACTION (No NLTK needed)
# =============================================================================
def extract_keywords_simple(text, num_keywords=15):
    """Extract keywords without NLTK - uses simple frequency analysis"""
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'who',
        'what', 'which', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'your', 'our', 'their', 'my', 'his', 'her', 'its', 'as', 'if', 'then',
        'because', 'while', 'although', 'after', 'before', 'above', 'below',
        'between', 'into', 'through', 'during', 'about', 'against', 'without'
    }
    
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_count = {}
    for word in words:
        if word not in stopwords:
            word_count[word] = word_count.get(word, 0) + 1
    
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:num_keywords]]

def generate_hashtags(keywords):
    """Generate hashtags from keywords"""
    hashtags = [f"#{kw.replace(' ', '')}" for kw in keywords[:10]]
    hashtags.extend(['#marketing', '#business', '#growth'])
    return list(set(hashtags))[:15]

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================
def get_tone_modifier(tone):
    """Get tone-specific instructions"""
    tones = {
        'Professional': "Use formal, business-appropriate language. Be authoritative.",
        'Emotional': "Connect emotionally. Use storytelling. Appeal to feelings.",
        'Exciting': "Use energetic, dynamic language. Create enthusiasm.",
        'Urgent': "Create urgency and scarcity. Use time-sensitive language.",
        'Friendly': "Use warm, conversational tone. Be approachable.",
        'Luxury': "Use sophisticated, premium language. Emphasize exclusivity."
    }
    return tones.get(tone, tones['Professional'])

def create_prompt(inputs):
    """Create the main content generation prompt"""
    return f"""You are an expert marketing copywriter. Generate marketing content for:

BUSINESS: {inputs['business_name']}
TYPE: {inputs['business_type']}
PRODUCT/SERVICE: {inputs['product_service']}
TARGET AUDIENCE: {inputs['target_audience']}
OFFER: {inputs['offer']}
TONE: {inputs['tone']} - {get_tone_modifier(inputs['tone'])}

Generate content in this exact JSON format:
{{
    "google_ads": {{
        "headlines": ["5 headlines, MAX 30 characters each"],
        "descriptions": ["3 descriptions, MAX 90 characters each"]
    }},
    "facebook": {{
        "primary_text": ["2 ad texts, 125-500 characters"],
        "headlines": ["3 headlines, MAX 40 characters"]
    }},
    "instagram": {{
        "captions": ["2 engaging captions with emojis"],
        "hashtags": ["15 relevant hashtags"]
    }},
    "seo": {{
        "titles": ["3 SEO titles, 50-60 characters"],
        "meta_descriptions": ["3 meta descriptions, 150-160 characters"]
    }},
    "landing_page": {{
        "headline": "Main headline",
        "subheadline": "Supporting text",
        "cta_buttons": ["3 CTA button texts"],
        "value_props": ["4 value propositions"]
    }},
    "keywords": ["10 target keywords"],
    "cta_suggestions": ["5 call-to-action phrases"]
}}

IMPORTANT: Return ONLY valid JSON. No markdown, no code blocks, no explanation."""

# =============================================================================
# CONTENT GENERATOR
# =============================================================================
def generate_content(api_key, inputs):
    """Generate content using OpenAI"""
    openai = load_openai()
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a marketing expert. Always respond with valid JSON only."},
                {"role": "user", "content": create_prompt(inputs)}
            ],
            max_tokens=3000,
            temperature=0.7
        )
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith('```'):
            content = re.sub(r'^```json?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        
        return json.loads(content)
    
    except json.JSONDecodeError:
        st.error("Failed to parse AI response. Please try again.")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================
def export_to_docx(content, inputs):
    """Export to Word document"""
    Document, Pt, WD_ALIGN_PARAGRAPH = load_docx()
    
    doc = Document()
    doc.add_heading('AI Generated Marketing Content', 0)
    
    doc.add_heading('Business Information', level=1)
    doc.add_paragraph(f"Business: {inputs.get('business_name', '')}")
    doc.add_paragraph(f"Type: {inputs.get('business_type', '')}")
    doc.add_paragraph(f"Product/Service: {inputs.get('product_service', '')}")
    doc.add_paragraph(f"Audience: {inputs.get('target_audience', '')}")
    doc.add_paragraph(f"Offer: {inputs.get('offer', '')}")
    doc.add_paragraph(f"Tone: {inputs.get('tone', '')}")
    
    def add_section(title, data):
        doc.add_heading(title, level=1)
        if isinstance(data, dict):
            for key, value in data.items():
                doc.add_heading(key.replace('_', ' ').title(), level=2)
                if isinstance(value, list):
                    for item in value:
                        doc.add_paragraph(f"• {item}", style='List Bullet')
                else:
                    doc.add_paragraph(str(value))
        elif isinstance(data, list):
            for item in data:
                doc.add_paragraph(f"• {item}", style='List Bullet')
    
    if content:
        for section, data in content.items():
            add_section(section.replace('_', ' ').title(), data)
    
    doc.add_paragraph(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def export_to_pdf(content, inputs):
    """Export to PDF"""
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, getSampleStyleSheet, ParagraphStyle, colors, inch, A4 = load_reportlab()
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("AI Generated Marketing Content", styles['Title']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Business Information", styles['Heading1']))
    info = f"""
    <b>Business:</b> {inputs.get('business_name', '')}<br/>
    <b>Type:</b> {inputs.get('business_type', '')}<br/>
    <b>Product/Service:</b> {inputs.get('product_service', '')}<br/>
    <b>Audience:</b> {inputs.get('target_audience', '')}<br/>
    <b>Offer:</b> {inputs.get('offer', '')}<br/>
    <b>Tone:</b> {inputs.get('tone', '')}
    """
    story.append(Paragraph(info, styles['Normal']))
    story.append(Spacer(1, 20))
    
    def add_content(data, level=0):
        if isinstance(data, dict):
            for key, value in data.items():
                title = key.replace('_', ' ').title()
                story.append(Paragraph(title, styles['Heading2']))
                add_content(value, level + 1)
        elif isinstance(data, list):
            for item in data:
                text = str(item).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(f"• {text}", styles['Normal']))
        else:
            text = str(data).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(text, styles['Normal']))
        story.append(Spacer(1, 10))
    
    if content:
        add_content(content)
    
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.title("🚀 AI Sales Agent")
        st.markdown("---")
        
        page = st.radio(
            "Menu",
            ["🏠 Home", "✨ Generate", "📊 History", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get('api_key', ''))
        if api_key:
            st.session_state['api_key'] = api_key
            st.success("✅ API Key set")
        
        return page

def render_home():
    """Home page"""
    st.title("🚀 AI Sales Copy & Ad Content Agent")
    st.markdown("### Generate High-Converting Marketing Content")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 Google Ads")
        st.markdown("Headlines & descriptions with character limits")
    
    with col2:
        st.markdown("#### 📱 Social Media")
        st.markdown("Facebook & Instagram content with hashtags")
    
    with col3:
        st.markdown("#### 🔍 SEO")
        st.markdown("Titles, meta descriptions & keywords")
    
    st.markdown("---")
    st.info("👈 Enter your OpenAI API key in the sidebar and click 'Generate' to start!")

def render_generate():
    """Generate content page"""
    st.title("✨ Generate Marketing Content")
    
    if not st.session_state.get('api_key'):
        st.warning("⚠️ Enter your OpenAI API key in the sidebar first!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        business_name = st.text_input("Business Name *", placeholder="TechFlow Solutions")
        business_type = st.selectbox("Business Type *", [
            "E-commerce", "SaaS", "Local Service", "Consulting", "Healthcare",
            "Education", "Real Estate", "Finance", "Restaurant", "Fitness", "Other"
        ])
        product_service = st.text_area("Product/Service *", placeholder="Describe your product...", height=100)
    
    with col2:
        target_audience = st.text_area("Target Audience *", placeholder="Who is your customer?", height=80)
        offer = st.text_input("Current Offer", placeholder="50% off for first 100 customers")
        tone = st.selectbox("Tone *", ["Professional", "Emotional", "Exciting", "Urgent", "Friendly", "Luxury"])
    
    platform = st.multiselect("Platforms", ["All Platforms", "Google Ads", "Facebook", "Instagram", "SEO", "Landing Page"], default=["All Platforms"])
    
    is_valid = all([business_name, business_type, product_service, target_audience])
    
    if st.button("🚀 Generate Content", type="primary", disabled=not is_valid, use_container_width=True):
        inputs = {
            'business_name': business_name,
            'business_type': business_type,
            'product_service': product_service,
            'target_audience': target_audience,
            'offer': offer,
            'tone': tone,
            'platform': platform
        }
        
        with st.spinner("🔄 Generating content... Please wait..."):
            results = generate_content(st.session_state['api_key'], inputs)
        
        if results:
            st.session_state['results'] = results
            st.session_state['inputs'] = inputs
            
            conn = init_database()
            save_to_history(conn, inputs, results)
            
            text = f"{product_service} {target_audience} {offer}"
            keywords = extract_keywords_simple(text)
            st.session_state['keywords'] = keywords
    
    if 'results' in st.session_state and st.session_state['results']:
        st.markdown("---")
        st.success("✅ Content Generated!")
        
        results = st.session_state['results']
        
        if 'keywords' in st.session_state:
            with st.expander("🔑 Extracted Keywords", expanded=True):
                st.write(", ".join(st.session_state['keywords']))
                st.write("**Hashtags:** " + " ".join(generate_hashtags(st.session_state['keywords'])))
        
        for section, content in results.items():
            with st.expander(f"📌 {section.replace('_', ' ').title()}", expanded=True):
                if isinstance(content, dict):
                    for key, value in content.items():
                        st.markdown(f"**{key.replace('_', ' ').title()}:**")
                        if isinstance(value, list):
                            for i, item in enumerate(value, 1):
                                st.markdown(f"{i}. {item}")
                        else:
                            st.write(value)
                elif isinstance(content, list):
                    for i, item in enumerate(content, 1):
                        st.markdown(f"{i}. {item}")
                else:
                    st.write(content)
        
        st.markdown("---")
        st.subheader("📥 Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            docx_file = export_to_docx(results, st.session_state['inputs'])
            st.download_button("📄 Download DOCX", docx_file, f"content_{datetime.now().strftime('%Y%m%d')}.docx")
        
        with col2:
            pdf_file = export_to_pdf(results, st.session_state['inputs'])
            st.download_button("📕 Download PDF", pdf_file, f"content_{datetime.now().strftime('%Y%m%d')}.pdf")
        
        with col3:
            st.download_button("📋 Download JSON", json.dumps(results, indent=2), f"content_{datetime.now().strftime('%Y%m%d')}.json")

def render_history():
    """History page"""
    st.title("📊 Content History")
    
    conn = init_database()
    history = get_history(conn)
    
    if not history:
        st.info("No content generated yet!")
        return
    
    st.metric("Total Generated", len(history))
    
    for record in history[:10]:
        with st.expander(f"📌 {record['business_name']} - {record['created_at'][:10] if record['created_at'] else 'N/A'}"):
            st.markdown(f"**Type:** {record['business_type']}")
            st.markdown(f"**Tone:** {record['tone']}")
            st.markdown(f"**Platform:** {record['platform']}")
            
            if record['full_response']:
                try:
                    st.json(json.loads(record['full_response']))
                except:
                    st.text(record['full_response'][:500])

def render_settings():
    """Settings page"""
    st.title("⚙️ Settings")
    
    api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.get('api_key', ''))
    if api_key:
        st.session_state['api_key'] = api_key
        st.success("API Key saved!")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History"):
        conn = init_database()
        conn.execute("DELETE FROM content_history")
        conn.commit()
        st.success("History cleared!")

# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main app"""
    st.markdown("""
    <style>
    .stButton>button {border-radius: 10px; height: 45px; font-weight: bold;}
    .stTextInput>div>div>input {border-radius: 8px;}
    .stTextArea>div>div>textarea {border-radius: 8px;}
    </style>
    """, unsafe_allow_html=True)
    
    init_database()
    
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = os.environ.get('OPENAI_API_KEY', '')
    
    page = render_sidebar()
    
    if page == "🏠 Home":
        render_home()
    elif page == "✨ Generate":
        render_generate()
    elif page == "📊 History":
        render_history()
    elif page == "⚙️ Settings":
        render_settings()

if __name__ == "__main__":
    main()
