# README
This is a simple Q&A agent made with PydanticAI and mock vector and structure search.  
## Dependencies  
The LLM is deployed through LM Studio, through which you should be able to run with any tool-calling model deployed the same way, although it was created and tested with **openai/gpt-oss-20b**.
It was initially made on WSL machine with an AMD card and an open-source model.  
The agent itself works on PydanticAI and it is deployed through FastAPI and Uvicorn.  
Sentence Transformers with ***all-MiniLM-L6-v2*** is used for mock vector search.  
## Implementation Overview
The chat history is implemented in-memory using FastAPI app states and PydanticAI message tracking.  
The same message tracking is used for response tracing. Although I have cleaned it up a bit for better readability..  
For a mock product API it uses a simple tool that takes a required month and year as integers, multiplies them together, and multiplies by 100 to return sales data.  
The agent handles unstructured data by embedding texts and user queries into a shared vector space using SentenceTransformer, then retrieving the top-matching results based on cosine similarity, including their text, metadata, and similarity score.
## How to Run and Test
You can run the image with ```docker run --rm -p 8000:8000 -e MODEL_NAME="openai/gpt-oss-20b" -e PROVIDER_URL="http://10.2.0.2:1234/v1" --name qa-agent qa-agent:latest```. Change the ***MODEL_NAME*** and ***PROVIDER_URL*** enviromental variables accordingly.  
After that try running a few command like that:
```curl -X POST "http://localhost:8000/query_agent" -H "Content-Type: application/json" -d '{"question": "Tell me more about our herbal tea?"}'```
```curl -X POST "http://localhost:8000/query_agent" -H "Content-Type: application/json" -d '{"question": "Hello, my name is John?"}'```
```curl -X POST "http://localhost:8000/query_agent" -H "Content-Type: application/json" -d '{"question": "What is my name"}'```
```curl -X POST "http://localhost:8000/query_agent" -H "Content-Type: application/json" -d '{"question": "How much money we made last month"}'```
The output should look like this:   
```{"output":"The sales for January 2024 were **$202,400**.","trace":[{"content":"What are the sales for January 2024?","timestamp":"2025-09-28T15:29:28.303061Z","part_kind":"user-prompt"},{"tool_name":"get_sales","args":"{\"year\":2024,\"month\":1}","tool_call_id":"504469511","part_kind":"tool-call"},{"tool_name":"get_sales","content":{"year":2024,"month":1,"sales":202400},"tool_call_id":"504469511","metadata":null,"timestamp":"2025-09-28T15:29:29.656590Z","part_kind":"tool-return"}],"usage":{"input_tokens":616,"cache_write_tokens":0,"cache_read_tokens":0,"output_tokens":71,"input_audio_tokens":0,"cache_audio_read_tokens":0,"output_audio_tokens":0,"details":{},"requests":2,"tool_calls":1}}```  
Which includes both traces, as well as usage statistics.  
