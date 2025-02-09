from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for Gemini
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    max_output_tokens=1000,
    temperature=0,
    convert_system_message_to_human=True
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create chain
chain = prompt | llm

# Session storage
store = {}

def get_chat_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Pydantic models
class UserRequest(BaseModel):
    input: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str

# API endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: UserRequest):
    try:
        response = chain_with_history.invoke(
            {"input": request.input},
            config={"configurable": {"session_id": request.session_id}}
        )
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    history = store.get(session_id, None)
    if not history:
        return {"history": []}
    return {
        "history": [
            {"type": message.type, "content": message.content}
            for message in history.messages
        ]
    }

@app.get("/api/all_sessions")
async def get_all_sessions():
    return {
        "sessions": {
            session_id: [
                {"type": message.type, "content": message.content}
                for message in history.messages
            ] for session_id, history in store.items()
        }
    }

# New endpoint to delete a session
@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    print(session_id)
    if session_id in store:
        print("store session", session_id)
        del store[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)