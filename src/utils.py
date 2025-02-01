import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('capstone_project.log')
    ]
)

# Create logger for this module
log = logging.getLogger(__name__)