import time
import json
import os
from datetime import date
import google.generativeai as genai
from config import GEMINI_MODEL, GEMINI_RPM_LIMIT, GEMINI_SAFE_DAILY_LIMIT, GEMINI_CALLS_LOG


class DailyCallTracker:
    def __init__(self, limit: int = GEMINI_SAFE_DAILY_LIMIT, log_path: str = GEMINI_CALLS_LOG):
        self.limit = limit
        self.log_path = log_path

    def _load(self) -> dict:
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                return json.load(f)
        return {}

    def _save(self, data: dict):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(data, f)

    def check_and_increment(self):
        today = str(date.today())
        data = self._load()
        count = data.get(today, 0)
        if count >= self.limit:
            raise RuntimeError(
                f"Gemini daily call limit reached ({count}/{self.limit}). Stopping to protect quota."
            )
        data[today] = count + 1
        self._save(data)

    def today_count(self) -> int:
        today = str(date.today())
        return self._load().get(today, 0)


class GeminiGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self._last_call = 0.0
        self._min_interval = 60.0 / GEMINI_RPM_LIMIT  # 4 s between calls
        self._tracker = DailyCallTracker()

    def generate(self, prompt: str) -> str:
        self._tracker.check_and_increment()
        elapsed = time.perf_counter() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        response = self.model.generate_content(prompt)
        self._last_call = time.perf_counter()
        return response.text
