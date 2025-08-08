# Configure logging
import logging, sys, os


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.abspath(os.path.join(MODULE_DIR, '../../logs'))
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, 'pdf_extraction.log')


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Create a logger
logger = logging.getLogger(__name__)

# Set the console handler encoding to handle Unicode characters
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setStream(sys.stdout)

# Add emojis to logging messages
logger.info("✨ Logging configured successfully.")

def test_logging():
    logger.error("🛠️ Testing logs.")
    logger.info("🛠️ Testing logs.")
    logger.debug("🛠️ Testing logs.")
    logger.critical("🛠️ Testing logs.")

if __name__ == "__main__":
    test_logging()