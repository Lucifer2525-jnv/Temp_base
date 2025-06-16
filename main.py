from fastapi import FastAPI, Request, Depends, HTTPException, Header
from pydantic import BaseModel
from backend.rag_agent import create_agent_with_memory
from backend.sqlite_memory import get_all_sessions, get_chat_history, clear_session, SQLiteMemory
import os
import uvicorn

app = FastAPI()

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "supersecretkey")

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    confidence: float

def verify_admin_key(x_api_key: str = Header(...)):
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    print(f"************* Processing question: {req.question} ************")
    try:
        memory = SQLiteMemory(req.session_id)
        
        # Fix: Unpack the tuple returned by create_agent_with_memory
        agent, db, embedding_model = create_agent_with_memory(session_id=req.session_id)
        print("************* Agent Created Successfully ************")
        
        # Get response from agent
        response = agent.run(req.question)
        print("Type of result from agent.run():", type(response))
        
        # Extract answer text from response
        if hasattr(response, 'content'):
            answer_text = response.content
        elif hasattr(response, 'text'):
            answer_text = response.text
        elif isinstance(response, dict):
            answer_text = response.get('answer', response.get('output', str(response)))
        else:
            answer_text = str(response)
        
        answer_text = str(answer_text).strip()

        # Calculate confidence based on similarity scores
        query_vector = embedding_model.embed_query(req.question)
        docs_and_scores = db.similarity_search_with_score_by_vector(query_vector, k=4)
        
        # Save to memory
        memory.save_context({"question": req.question}, {"answer": answer_text})
       
        # Calculate confidence
        similarities = [score for _, score in docs_and_scores]
        confidence = 1 - (sum(similarities) / len(similarities)) if similarities else 0.5
        
        return ChatResponse(
            answer=answer_text,
            confidence=round(confidence * 100, 2)
        )

    except Exception as e:
        print(f"Error in chat_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/list_sessions", dependencies=[Depends(verify_admin_key)])
def list_sessions():
    return {"sessions": get_all_sessions()}

@app.get("/chat_history/{session_id}", dependencies=[Depends(verify_admin_key)])
def session_history(session_id: str):
    history = get_chat_history(session_id)
    return {"history": [{"role": r, "message": m} for r, m in history]}

@app.post("/reset_session/{session_id}", dependencies=[Depends(verify_admin_key)])
def reset_session(session_id: str):
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
