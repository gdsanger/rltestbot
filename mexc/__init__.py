"""MEXC utilities for reinforcement learning."""

from dotenv import load_dotenv, find_dotenv

# Load .env from project root so API keys are available
load_dotenv(find_dotenv(), override=False)
