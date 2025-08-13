import os
from dotenv import load_dotenv
import google.generativeai as genai
import time
import requests
import json

load_dotenv()

def process_transcript(transcript):
    providers = [
        ("google_gemini", try_google_gemini),
        ("openai_gpt", try_openai_gpt),
        ("anthropic_claude", try_anthropic_claude),
        ("huggingface", try_huggingface),
        ("demo", process_transcript_demo)
    ]
    
    for provider_name, provider_func in providers:
        try:
            print(f"Trying {provider_name}...")
            result = provider_func(transcript)
            if result and not is_quota_error(result):
                print(f"Success with {provider_name}")
                return result
            elif is_quota_error(result):
                print(f"Quota exceeded with {provider_name}, trying next...")
                continue
        except Exception as e:
            print(f"Error with {provider_name}: {str(e)}")
            continue
    
    return process_transcript_demo(transcript)

def is_quota_error(result):

    if not result:
        return False
    quota_indicators = ["quota", "exceeded", "429", "rate limit", "limit exceeded"]
    return any(indicator in str(result).lower() for indicator in quota_indicators)

def try_google_gemini(transcript):

    api_keys = get_multiple_api_keys("GOOGLE_API_KEY")
    
    for i, api_key in enumerate(api_keys):
        if not api_key:
            continue
            
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            
            prompt = f"""
You are a helpful meeting assistant. Analyze the following meeting transcript and return:

1. A concise summary of what was discussed.
2. A list of key decisions made.
3. A list of tasks or action items (with who is responsible and deadline if mentioned).

Transcript:
{transcript}

Respond with clearly separated sections:
- Summary
- Decisions
- Action Items

Format the response nicely with clear headings and bullet points.
"""
            
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text
                
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg and "quota" in error_msg.lower():
                print(f" API key {i+1} quota exceeded, trying next...")
                continue
            else:
                print(f"Error with API key {i+1}: {error_msg}")
                continue
    
    return "All Google API keys have quota exceeded. Trying other providers..."

def try_openai_gpt(transcript):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful meeting assistant. Analyze meeting transcripts and provide summaries, decisions, and action items."
                },
                {
                    "role": "user",
                    "content": f"Analyze this meeting transcript:\n\n{transcript}\n\nProvide:\n1. Summary\n2. Key Decisions\n3. Action Items"
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        elif response.status_code == 429:
            return "OpenAI rate limit exceeded"
        else:
            return f"OpenAI error: {response.status_code}"
            
    except Exception as e:
        return f"OpenAI error: {str(e)}"

def try_anthropic_claude(transcript):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    
    try:
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze this meeting transcript:\n\n{transcript}\n\nProvide:\n1. Summary\n2. Key Decisions\n3. Action Items"
                }
            ]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["content"][0]["text"]
        elif response.status_code == 429:
            return "Anthropic rate limit exceeded"
        else:
            return f"Anthropic error: {response.status_code}"
            
    except Exception as e:
        return f" Anthropic error: {str(e)}"

def try_huggingface(transcript):
    
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        return None
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": f"Summarize this meeting transcript:\n\n{transcript}\n\nProvide summary, decisions, and action items.",
            "parameters": {
                "max_length": 500,
                "temperature": 0.7
            }
        }
        
        response = requests.post(
            "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0]["summary_text"]
            else:
                return str(result)
        elif response.status_code == 429:
            return "Hugging Face rate limit exceeded"
        else:
            return f" Hugging Face error: {response.status_code}"
            
    except Exception as e:
        return f"Hugging Face error: {str(e)}"

def get_multiple_api_keys(base_key_name):

    keys = []
    
   
    main_key = os.getenv(base_key_name)
    if main_key:
        keys.append(main_key)
    
    i = 1
    while True:
        additional_key = os.getenv(f"{base_key_name}_{i}")
        if additional_key:
            keys.append(additional_key)
            i += 1
        else:
            break
    
    return keys

def handle_quota_error(transcript):

    return f"""


**What happened:**
You've reached the free tier limit for Google Gemini API. This is a common issue with free accounts.

**Immediate Solutions:**



{process_transcript_demo(transcript)}

def process_transcript_demo(transcript):
   
    if not transcript.strip():
        return "Please provide a transcript to summarize."
    
    words = transcript.split()
    word_count = len(words)
    
    
    demo_summary = f"""
**Transcript Length:** {word_count} words

**Summary:**
This is a demo summary of your meeting transcript. The actual AI-powered summarization requires a valid API key and available quota.

**Key Points:**
- Meeting transcript contains {word_count} words
- To get real AI summarization, ensure your API key is valid and has available quota
- Current mode: Demo/Testing due to quota limits

**Action Items:**
- Check your API key status
- Consider using multiple API keys or alternative providers
- The AI will then provide detailed meeting analysis

    
    return demo_summary

def process_transcript_with_retry(transcript, max_retries=3, delay=60):
    for attempt in range(max_retries):
        try:
            result = process_transcript(transcript)
            

            if is_quota_error(result):
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    return result
            else:
                return result
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
            else:
                return f"Error after {max_retries} attempts: {str(e)}"
    
    return "Maximum retry attempts reached." 

    

