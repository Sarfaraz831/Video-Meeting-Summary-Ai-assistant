#Actionableitems , decision , questions 
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os 


def get_llm():
    # return ChatGoogleGenerativeAI(
    #     model = "gemini-2.5-flash",
    #     google_api_key = os.getenv("GOOGLE_API_KEY"),
    #     temperature = 0.2,
    #     max_retries=5
    # )

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature = 0.2,
        max_retries=5
    )


def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )


def extract_all_metadata(transcript: str) -> dict:
    """Extract action items, key decisions, and open questions in a single LLM call to save tokens and avoid rate limits."""
    print("Extracting action items, key decisions, and open questions (consolidated)...")
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, extract the following three sections:\n\n"
        "1. Action Items:\n"
        "Extract all action items. For each provide a task description, owner (responsible party), and deadline (if mentioned, else 'Not specified'). Format as a numbered list.\n\n"
        "2. Key Decisions:\n"
        "Extract all key decisions made. Format as a numbered list.\n\n"
        "3. Open Questions:\n"
        "Extract all unresolved questions or topics needing follow-up. Format as a numbered list.\n\n"
        "Format your output exactly as follows with no other intro or outro text. Ensure you use the exact separator tags:\n"
        "===ACTION_ITEMS===\n"
        "[Action items list here, or 'No action items found.']\n"
        "===DECISIONS===\n"
        "[Key decisions list here, or 'No key decisions found.']\n"
        "===QUESTIONS===\n"
        "[Open questions list here, or 'No open questions found.']"
    )
    
    response = chain.invoke(transcript)
    
    # Parse sections
    action_items = "No action items found."
    decisions = "No key decisions found."
    questions = "No open questions found."
    
    try:
        parts = response.split("===ACTION_ITEMS===")
        if len(parts) > 1:
            subparts = parts[1].split("===DECISIONS===")
            action_items = subparts[0].strip()
            if len(subparts) > 1:
                subsubparts = subparts[1].split("===QUESTIONS===")
                decisions = subsubparts[0].strip()
                if len(subsubparts) > 1:
                    questions = subsubparts[1].strip()
    except Exception as e:
        print(f"Warning: Error parsing consolidated metadata: {e}. Returning raw output.")
        # Fallback if structure is slightly malformed
        return {
            "action_items": response,
            "key_decisions": "Extracted with action items",
            "open_questions": "Extracted with action items"
        }
                
    return {
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions
    }


def extract_action_items(transcript:str)->str:
    return extract_all_metadata(transcript)["action_items"]


def extract_key_decisions(transcript: str) -> str:
    return extract_all_metadata(transcript)["key_decisions"]


def extract_questions(transcript: str) -> str:
    return extract_all_metadata(transcript)["open_questions"]