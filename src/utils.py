import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('capstone_project.log'),
        logging.StreamHandler()
    ]
)

# Create logger for this module
log = logging.getLogger(__name__)