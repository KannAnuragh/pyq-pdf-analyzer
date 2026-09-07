"""
Google Gemini LLM Integration.

Handles streaming and non-streaming responses using google-genai.
Includes token usage tracking for observability.
"""

import os
from typing import AsyncGenerator, Optional
from google import genai
from google.genai import types

from core.config import settings
from llm.prompts import SYSTEM_PROMPT
from providers.base import BaseLLMProvider


class GeminiClient(BaseLLMProvider):
    """Singleton wrapper for Google Gemini API."""
    _instance = None
    _client = None
    _current_api_key = None
    _last_error = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GeminiClient, cls).__new__(cls)
        return cls._instance

    def _get_client(self):
        # Retrieve fresh key from env or settings
        raw_key = os.environ.get("GEMINI_API_KEY") or settings.GEMINI_API_KEY or ""
        api_key = raw_key.strip()
        
        if not api_key:
            self._client = None
            self._current_api_key = ""
            self._last_error = "GEMINI_API_KEY is empty in backend/.env"
            return None

        # Re-initialize client if key changed or client is None
        if self._client is None or self._current_api_key != api_key:
            try:
                self._client = genai.Client(api_key=api_key)
                self._current_api_key = api_key
                self._last_error = None
            except Exception as e:
                print(f"Error initializing genai.Client: {e}")
                self._client = None
                self._last_error = str(e)
                return None

        return self._client
        
    def _get_config(
        self,
        system_instruction: Optional[str] = None,
        response_mime_type: Optional[str] = None,
        max_output_tokens: int = 8192,
    ) -> types.GenerateContentConfig:
        """Get standard generation config."""
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
            # Hard stop the instant the model tries to fabricate the next
            # student turn itself (seen in quiz mode: it would grade its own
            # invented "Student Answer: X" and keep going through several
            # questions unprompted). A prompt instruction alone wasn't
            # reliable enough to prevent this — this is a hard boundary.
            stop_sequences=["Student Answer:"],
        )

    async def stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response chunks from Gemini asynchronously."""
        client = self._get_client()
        if not client:
            error_details = self._last_error or "GEMINI_API_KEY is missing or invalid."
            yield f"\n\n[Gemini API Error: {error_details}\n\nPlease ensure you are using a valid Google AI Studio API Key (starting with 'AIzaSy...') in your `backend/.env` file and restart the backend container using `docker compose restart backend`.]"
            return
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🤖 [GEMINI API CALL] Model: {settings.LLM_MODEL} | Prompt length: {len(prompt)} chars (~{len(prompt) // 4} tokens) (Attempt {attempt+1}/{max_retries})", flush=True)
                
                response_stream = await client.aio.models.generate_content_stream(
                    model=settings.LLM_MODEL,
                    contents=prompt,
                    config=self._get_config(system_instruction=SYSTEM_PROMPT)
                )
                
                total_output_chars = 0
                last_chunk = None
                async for chunk in response_stream:
                    if chunk.text:
                        total_output_chars += len(chunk.text)
                        yield chunk.text
                    last_chunk = chunk
                
                # Extract and log token usage from the final chunk's usage_metadata
                if last_chunk and hasattr(last_chunk, 'usage_metadata') and last_chunk.usage_metadata:
                    usage = last_chunk.usage_metadata
                    input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
                    output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
                    total_tokens = getattr(usage, 'total_token_count', 0) or (input_tokens + output_tokens)
                    
                    print(f"📊 [GEMINI TOKEN USAGE]", flush=True)
                    print(f"   Input tokens:  {input_tokens}", flush=True)
                    print(f"   Output tokens: {output_tokens}", flush=True)
                    print(f"   Total tokens:  {total_tokens}", flush=True)
                else:
                    # Fallback to character-based estimation
                    input_tokens_est = len(prompt) // 4
                    output_tokens_est = total_output_chars // 4
                    print(f"📊 [GEMINI TOKEN USAGE (estimated)]", flush=True)
                    print(f"   Input:  ~{input_tokens_est} tokens ({len(prompt)} chars)", flush=True)
                    print(f"   Output: ~{output_tokens_est} tokens ({total_output_chars} chars)", flush=True)
                    print(f"   Total:  ~{input_tokens_est + output_tokens_est} tokens", flush=True)
                
                # Success, break retry loop
                break

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Too Many Requests" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < max_retries - 1:
                        import asyncio
                        wait_seconds = (attempt + 1) * 3
                        print(f"⚠️ [GEMINI 429 RATE LIMIT] Rate limit hit. Retrying in {wait_seconds}s... (Attempt {attempt+1}/{max_retries})", flush=True)
                        await asyncio.sleep(wait_seconds)
                        continue
                    else:
                        print(f"❌ [GEMINI 429 RATE LIMIT EXCEEDED] Max retries reached for 429: {e}", flush=True)
                        yield "\n\n[Google Gemini Rate Limit Exceeded (429): You have exceeded the free tier quota for this model. Please wait a few seconds before asking another question, or switch to `gemini-2.0-flash-lite` or `gemini-1.5-flash` in `backend/.env`.]"
                        return
                elif "404" in err_str or "Not Found" in err_str:
                    print(f"❌ [GEMINI 404 MODEL NOT FOUND] Invalid model name '{settings.LLM_MODEL}': {e}", flush=True)
                    yield f"\n\n[Google Gemini Error (404 Not Found): The model name `{settings.LLM_MODEL}` does not exist in Google AI Studio. Please check `LLM_MODEL` in your `backend/.env` file and set it to a valid model such as `gemini-2.0-flash`, `gemini-2.0-flash-lite`, or `gemini-1.5-flash`.]"
                    return
                elif "Expecting property name" in err_str or "JSONDecodeError" in err_str or "char 1" in err_str:
                    print(f"❌ [GEMINI API KEY ERROR] Invalid API response from Gemini: {e}", flush=True)
                    yield f"\n\n[Google Gemini API Error: The API returned an invalid response (non-JSON). This typically occurs when `GEMINI_API_KEY` in `backend/.env` is invalid or expired. Please verify your Google AI Studio API Key (which starts with `AIzaSy...`) in `backend/.env`, set `LLM_MODEL=gemini-2.0-flash`, and restart the backend.]"
                    return
                else:
                    print(f"Gemini Streaming Error: {e}")
                    yield f"\n\n[Error generating response: {err_str}]"
                    return

    def generate_response(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_mime_type: Optional[str] = None,
    ) -> str:
        """Generate a complete response synchronously."""
        client = self._get_client()
        if not client:
            error_details = self._last_error or "GEMINI_API_KEY is missing or invalid."
            return f"Error: {error_details}. Please ensure you are using a valid Google AI Studio API Key starting with 'AIzaSy...' in backend/.env."
        
        # Auto-enforce JSON mode when prompt explicitly asks for strict JSON extraction/quiz structure
        if response_mime_type is None and any(
            kw in prompt for kw in ["JSON object", "JSON shape", "strict JSON", "EXACTLY this JSON", "quiz-generation tool", "data-extraction tool"]
        ):
            response_mime_type = "application/json"

        try:
            response = client.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=self._get_config(
                    system_instruction=system_instruction,
                    response_mime_type=response_mime_type,
                )
            )
            
            # Log token usage
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                print(f"📊 [GEMINI TOKEN USAGE] Input: {getattr(usage, 'prompt_token_count', '?')} | Output: {getattr(usage, 'candidates_token_count', '?')} | Total: {getattr(usage, 'total_token_count', '?')}", flush=True)
            
            return response.text or ""
        except Exception as e:
            err_str = str(e)
            if "Expecting property name" in err_str or "JSONDecodeError" in err_str or "char 1" in err_str:
                return "Error: Invalid response from Gemini API. Please check GEMINI_API_KEY (starts with 'AIzaSy...') and LLM_MODEL in backend/.env."
            print(f"Gemini Error: {e}")
            return f"Error generating response: {str(e)}"

    async def generate_response_async(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_mime_type: Optional[str] = None,
    ) -> str:
        """Generate a complete response asynchronously using native async API."""
        client = self._get_client()
        if not client:
            error_details = self._last_error or "GEMINI_API_KEY is missing or invalid."
            return f"Error: {error_details}"
        
        # Auto-enforce JSON mode when prompt explicitly asks for strict JSON extraction/quiz structure
        if response_mime_type is None and any(
            kw in prompt for kw in ["JSON object", "JSON shape", "strict JSON", "EXACTLY this JSON", "quiz-generation tool", "data-extraction tool"]
        ):
            response_mime_type = "application/json"

        try:
            response = await client.aio.models.generate_content(
                model=settings.LLM_MODEL,
                contents=prompt,
                config=self._get_config(
                    system_instruction=system_instruction,
                    response_mime_type=response_mime_type,
                )
            )
            
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                print(f" [GEMINI ASYNC TOKEN USAGE] Input: {getattr(usage, 'prompt_token_count', '?')} | Output: {getattr(usage, 'candidates_token_count', '?')} | Total: {getattr(usage, 'total_token_count', '?')}", flush=True)
            
            return response.text or ""
        except Exception as e:
            err_str = str(e)
            if "Expecting property name" in err_str or "JSONDecodeError" in err_str or "char 1" in err_str:
                return "Error: Invalid response from Gemini API. Please check GEMINI_API_KEY (starts with 'AIzaSy...') and LLM_MODEL in backend/.env."
            print(f"Gemini Async Error: {e}")
            return f"Error generating response: {str(e)}"


# Global singleton instance
gemini_client = GeminiClient()