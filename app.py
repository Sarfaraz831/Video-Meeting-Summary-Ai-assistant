import streamlit as st
from main import run_pipeline
from core.rag_engine import ask_question
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥Video / Meeting Summary AI Assistant")
st.write("Upload a meeting recording or paste a video link to generate notes and chat with it.")

# ---------------- Sidebar ----------------

st.sidebar.header("Input")

language = st.sidebar.selectbox(
    "Summary Language",
    ["english", "hinglish"]
)

source = st.sidebar.text_input(
    "Video URL OR Local File Path"
)

process_btn = st.sidebar.button("🚀 Process")

# ---------------- Main ----------------

if process_btn:

    if source == "":
        st.warning("Please enter a video path or Video URL.")
        st.stop()

    with st.spinner("Processing video... This may take few minutes."):

        result = run_pipeline(source, language)

    st.session_state["result"] = result

    st.success("Processing Complete!")

# ---------------- Show Results ----------------

if "result" in st.session_state:

    result = st.session_state["result"]

     # ---------------- PDF Report ----------------

    styles = getSampleStyleSheet()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)

    story = []

    story.append(Paragraph("<b>AI Meeting Report</b>", styles["Title"]))
    story.append(Paragraph("<br/>", styles["BodyText"]))

    story.append(Paragraph("<b>Meeting Title</b>", styles["Heading2"]))
    story.append(Paragraph(result["title"], styles["BodyText"]))

    story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
    story.append(
        Paragraph(result["summary"].replace("\n", "<br/>"), styles["BodyText"])
    )

    story.append(Paragraph("<b>Transcript</b>", styles["Heading2"]))
    story.append(
        Paragraph(result["transcript"].replace("\n", "<br/>"), styles["BodyText"])
    )

    story.append(Paragraph("<b>Action Items</b>", styles["Heading2"]))
    story.append(
        Paragraph(result["action_items"].replace("\n", "<br/>"), styles["BodyText"])
    )

    story.append(Paragraph("<b>Key Decisions</b>", styles["Heading2"]))
    story.append(
        Paragraph(result["key_decisions"].replace("\n", "<br/>"), styles["BodyText"])
    )

    story.append(Paragraph("<b>Open Questions</b>", styles["Heading2"]))
    story.append(
        Paragraph(result["open_questions"].replace("\n", "<br/>"), styles["BodyText"])
    )

    if "chat_history" in st.session_state and st.session_state.chat_history:

        story.append(Paragraph("<b>Chat History</b>", styles["Heading2"]))

        for sender, message in st.session_state.chat_history:
            story.append(
                Paragraph(
                    f"<b>{sender}:</b> {message.replace(chr(10), '<br/>')}",
                    styles["BodyText"],
                )
            )

    doc.build(story)
    buffer.seek(0)

    st.download_button(
        label="📄 Download Meeting Report (PDF)",
        data=buffer,
        file_name=f"{result['title'].replace(' ', '_')}_meeting_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


#--------Download Button Ending------

    st.header("📌 Meeting Title")
    st.info(result["title"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summary",
        "Transcript",
        "Tasks",
        "Questions",
        "Chat"
    ])

    # ---------------- Summary ----------------

    with tab1:

        st.subheader("📋 Summary")

        st.write(result["summary"])

    # ---------------- Transcript ----------------

    with tab2:

        st.subheader("📝 Transcript")

        st.text_area(
            "",
            result["transcript"],
            height=500
        )

    # ---------------- Tasks ----------------

    with tab3:

        st.subheader("✅ Action Items")

        st.write(result["action_items"])

        st.divider()

        st.subheader("🔑 Key Decisions")

        st.write(result["key_decisions"])

    # ---------------- Questions ----------------

    with tab4:

        st.subheader("❓ Open Questions")

        st.write(result["open_questions"])

    # ---------------- Chat ----------------

    with tab5:

        st.subheader("💬 Chat with your Meeting")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_question = st.chat_input("Ask anything about the meeting...")

        if user_question:

            answer = ask_question(
                result["rag_chain"],
                user_question
            )

            st.session_state.chat_history.append(
                ("You", user_question)
            )

            st.session_state.chat_history.append(
                ("Assistant", answer)
            )

        for sender, message in st.session_state.chat_history:

            with st.chat_message(
                "user" if sender == "You" else "assistant"
            ):
                st.markdown(message)
