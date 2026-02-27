import streamlit as st
from agent_brain import resume_agent

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Resume")
    resume_text = st.text_area("Paste Resume Text", height=300)

with col2:
    st.subheader("💼 Job Description")
    job_description = st.text_area("Paste Job Description", height=300)

st.markdown("---")

if st.button("🚀 Analyze Resume"):

    if not resume_text or not job_description:
        st.warning("Please enter both Resume and Job Description.")
    else:
        with st.spinner("Analyzing..."):
            result = resume_agent(resume_text, job_description)

        st.success("Analysis Complete!")

        st.subheader("📊 Match Score")
        st.progress(int(result["score"]))
        st.metric("Match %", f"{result['score']}%")
        st.metric("Recommendation", result["recommendation"])
        st.metric("Risk", result["risk"])
        st.metric("Confidence", f"{result['confidence']}%")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🟢 Matched Skills")
            for skill in result["matched"]:
                st.write(f"✔ {skill}")

        with col2:
            st.subheader("🔴 Missing Skills")
            for skill in result["missing"]:
                st.write(f"✘ {skill}")

        st.markdown("---")

        st.subheader("📂 Category Distribution")
        for cat, val in result["categories"].items():
            st.write(f"{cat}: {val}")

        st.markdown("---")

        st.subheader("📈 Resume Intelligence")
        st.write(f"Years of Experience: {result['years']}")
        st.write(f"Leadership Signals: {result['leadership']}")
        st.write(f"Impact Metrics Present: {result['impact']}")

        st.markdown("---")

        st.subheader("🤖 AI Evaluation")
        st.write(result["analysis"])