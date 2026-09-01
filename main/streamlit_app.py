import streamlit as st
from agent_brain import resume_agent
from file_parser import extract_text

st.set_page_config(
    page_title="Semantic ATS Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Semantic ATS Resume Analyzer")

with st.expander("🔒 Privacy — what happens to your data"):
    st.markdown(
        "- Nothing you upload or paste is stored. It only exists in memory "
        "for this session and is discarded when you close the tab.\n"
        "- Your **raw resume/JD text never leaves this app**. Only the "
        "extracted skill list and match score are sent to the AI evaluation "
        "step, via whichever LLM provider is configured "
        "([LiteLLM](https://github.com/BerriAI/litellm), Groq by default) "
        "-- and only if that step runs at all.\n"
        "- No accounts, cookies, or analytics are used by this app."
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume")
    resume_file = st.file_uploader(
        "Upload resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"]
    )
    resume_text_input = st.text_area(
        "...or paste resume text", height=250,
        help="Used only if no file is uploaded above."
    )

with col2:
    st.subheader("Job Description")
    job_description = st.text_area("Paste Job Description", height=300)

st.markdown("---")

if st.button("🚀 Analyze Resume"):

    resume_text = ""
    if resume_file is not None:
        try:
            resume_text = extract_text(resume_file)
        except Exception as e:
            st.error(f"Couldn't read the uploaded file: {e}")
    else:
        resume_text = resume_text_input

    if not resume_text or not job_description:
        st.warning("Please provide a resume (upload or paste) and a job description.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = resume_agent(resume_text, job_description)
            except Exception as e:
                st.error(f"Something went wrong during analysis: {e}")
                st.stop()

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
            if result["matched"]:
                for skill in result["matched"]:
                    st.write(f"✔ {skill.title()}")
            else:
                st.caption("No matching skills found.")

        with col2:
            st.subheader("🔴 Missing Skills")
            if result["missing"]:
                for skill in result["missing"]:
                    st.write(f"✘ {skill.title()}")
            else:
                st.caption("No missing skills — full coverage!")

        st.markdown("---")

        st.subheader("📂 Category Distribution")
        nonzero_categories = {cat: val for cat, val in result["categories"].items() if val > 0}
        if nonzero_categories:
            st.bar_chart(nonzero_categories)
        else:
            st.caption("No categorized skills among the matches.")

        st.markdown("---")

        st.subheader("📈 Resume Intelligence")
        st.write(f"Years of Experience: {result['years']}")
        st.write(f"Leadership Signals: {'Yes' if result['leadership'] else 'No'}")
        st.write(f"Impact Metrics Present: {'Yes' if result['impact'] else 'No'}")

        st.markdown("---")

        st.subheader("🤖 AI Evaluation")
        if result["analysis_error"]:
            st.warning(result["analysis"])
        else:
            st.write(result["analysis"])
