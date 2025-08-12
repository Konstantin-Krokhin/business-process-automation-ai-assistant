import os
import json
import logging
from typing import List, Dict, Any, Optional

import openai

from backend.app.tools.account_api import get_account_balance

openai.api_key = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger

SYSTEM_PROMPT = "You are a helpful support agent. Use provided docs and tools."

class LLMClient:
	def __init__(self, model: str = "gpt-3.5-turbo", max_tokens: int = 256, temperature: float = 0.2):
		self.model = model
		self.max_tokens = max_tokens
		self.temperature = temperature

	def call_llm(self, messages: List[Dict[str, str]]) -> str:
		try:
			response = openai.ChatCompletion.create(
				model=self.model,
				messages=messages,
				max_tokens=self.max_tokens,
				temperature=self.temperature,
			)
			return response.choices[0].message['content']
		except Exception as e:
			logger.error(f"OpenAI API error: {e}")
            return "Sorry, I am having trouble processing your request right now."

	def handle_user_query(self, user_id: str, user_query: str, retrieved_docs: List[tuple]) -> str:
		docs_text = "\n\n".join([f"Document {i+1}:\n{doc[0]}" for i, doc in retrieved_docs])

		messages = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "system", "content": "Context:\n" + docs_text},
			{"role": "user", "content": user_query},
		]

		llm_output = self.call_llm(messages)

		try:
			parsed = json.loads(llm_output)
			if parsed.get("tool") == "get_account_balance":
				args = parsed.get("args", {})
				tool_res = get_account_balance(args.get("user_id", user_id))
				messages.append({"role": "assistant", "content": llm_output})
				messages.append({"role": "user", "content": f"Tool result: {tool_res}"})
				final_response = self.call_llm(messages)
				return final_response
		except json.JSONDecodeError:
			logger.warning("LLM output is not valid JSON, returning raw output.")

		return llm_output

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    client = LLMClient()
    # Simulated docs
    docs = [("Your password can be reset via the account settings page.", 0.85)]
    response = client.handle_user_query("user123", "How do I reset my password?", docs)
    print(response)