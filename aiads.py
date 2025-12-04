import streamlit as st
import json
import re

st.set_page_config(page_title="AI Sales Agent", page_icon="🚀", layout="wide")

st.title("🚀 AI Sales Copy Generator")

# API Key
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.warning("👈 Enter your OpenAI API key in sidebar")
    st.stop()

# Form
st.subheader("Enter Business Details")

col1, col2 = st.columns(2)

with col1:
    business_name = st.text_input("Business Name", placeholder="TechFlow Solutions")
    business_type = st.selectbox("Business Type", ["E-commerce", "SaaS", "Service", "Consulting", "Healthcare", "Restaurant", "Other"])
    product = st.text_area("Product/Service", placeholder="What do you sell?", height=100)

with col2:
    audience = st.text_area("Target Audience", placeholder="Who is your customer?", height=80)
    offer = st.text_input("Current Offer", placeholder="50% off first order")
    tone = st.selectbox("Tone", ["Professional", "Friendly", "Exciting", "Urgent", "Luxury"])

# Generate
if st.button("🚀 Generate Content", type="primary", use_container_width=True):
    
    if not all([business_name, product, audience]):
        st.error("Fill all required fields!")
        st.stop()
    
    prompt = f"""Create marketing content for:
Business: {business_name} ({business_type})
Product: {product}
Audience: {audience}
Offer: {offer}
Tone: {tone}

Return ONLY this JSON:
{{"google_ads":{{"headlines":["5 headlines under 30 chars"],"descriptions":["3 descriptions under 90 chars"]}},"facebook":{{"posts":["2 ad posts"]}},"instagram":{{"captions":["2 captions with emojis"],"hashtags":["10 hashtags"]}},"seo":{{"titles":["3 SEO titles"],"meta_descriptions":["3 meta descriptions"]}},"ctas":["5 call-to-action phrases"]}}"""

    with st.spinner("⏳ Generating..."):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            
            text = response.choices[0].message.content.strip()
            
            # Clean JSON
            if "```" in text:
                text = re.sub(r'```json?\n?', '', text)
                text = re.sub(r'```', '', text)
            
            data = json.loads(text)
            
            st.success("✅ Content Generated!")
            st.divider()
            
            # Display results
            for section, content in data.items():
                st.subheader(f"📌 {section.replace('_', ' ').title()}")
                
                if isinstance(content, dict):
                    for key, values in content.items():
                        st.markdown(f"**{key.replace('_', ' ').title()}:**")
                        if isinstance(values, list):
                            for i, v in enumerate(values, 1):
                                st.write(f"{i}. {v}")
                        else:
                            st.write(values)
                elif isinstance(content, list):
                    for i, v in enumerate(content, 1):
                        st.write(f"{i}. {v}")
                
                st.divider()
            
            # Download
            st.download_button(
                "📥 Download JSON",
                json.dumps(data, indent=2),
                "marketing_content.json",
                "application/json"
            )
            
        except json.JSONDecodeError:
            st.error("Failed to parse response. Try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
