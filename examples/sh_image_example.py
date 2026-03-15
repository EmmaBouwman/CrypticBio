from src.data_gather import get_image, sh_image_path, check_exists_dir
from dotenv import load_dotenv
from sentinelhub import SHConfig
import os

load_dotenv(".env_sentinel")
config = SHConfig()
config.sh_client_id = os.getenv("SH_CLIENT_ID")
config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")

check_exists_dir(sh_image_path)
get_image(51.788106, 5.92189, "2022-3-17", config, sh_image_path / "2_2500.png", width=2500)
