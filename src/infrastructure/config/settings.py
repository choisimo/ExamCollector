from dotenv import load_dotenv
import os

load_dotenv()

# OCR provider options: local, aws_textract, google_vision, none
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "local")

# AWS Textract credentials
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Google Vision API key
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
# OCR.Space API key (free tier)
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

# OpenAI API key for AI agent
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Directory for AI agent config JSON files
AI_AGENT_CONFIG_DIR = os.getenv(
    "AI_AGENT_CONFIG_DIR",
    os.path.join(os.path.dirname(__file__), "ai_agent")
)
