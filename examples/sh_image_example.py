import os
from pathlib import Path

from dotenv import load_dotenv
from sentinelhub import SHConfig
from pathlib import Path
from src.data_gather import SentinelHubManager, check_exists_dir, build_db

# load_dotenv(".env_sentinel")
# load_dotenv(".env")
# config = SHConfig()
# config.sh_client_id = os.getenv("SH_CLIENT_ID")
# config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")

# sh = SentinelHubManager(config)

# base_path = Path(os.getenv("BASE_FOLDER"))
# sh_image_path = base_path / os.getenv("SENTINEL_IMAGE_PATH", "images_sh")
# check_exists_dir(sh_image_path)

# target_date = "2022-03-17"
# target_file = sh_image_path / "2_2500.png"
# sh.get_and_save_image(51.788106, 5.92189, target_date, target_file)

tmp_folder = Path("./tmp")
check_exists_dir(tmp_folder)
build_db(None, None, tmp_folder, tmp_folder, None)

