import db
import config
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def migrate():
    LOGGER.info("Starting schema migration to Horizon 2...")
    try:
        con = db.get_db_connection(config.DATA_DIR)
        
        # Check if table exists
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='protein_structures'")
        if cur.fetchone():
            LOGGER.info("? Table 'protein_structures' exists.")
        else:
            LOGGER.error("? Table 'protein_structures' missing after connection init.")
            
        con.close()
        LOGGER.info("Migration check complete.")
    except Exception as e:
        LOGGER.exception("Migration failed: %s", e)

if __name__ == "__main__":
    migrate()
