import importlib.util
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Dynamically import script 01_fetch_data
script_path = Path(__file__).parent.parent / "scripts" / "01_fetch_data.py"
spec = importlib.util.spec_from_file_location("fetch_data", script_path)
fetch_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_data)


def test_fetch_top_species_mocked():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "count": 500,
                "taxon": {"rank": "species", "name": "Danaus plexippus", "id": 48662}
            },
            {
                "count": 300,
                "taxon": {"rank": "species", "name": "Vanessa cardui", "id": 47910}
            },
            {
                "count": 50,  # Below min threshold
                "taxon": {"rank": "species", "name": "Small Species", "id": 10001}
            }
        ]
    }

    with patch("requests.get", return_value=mock_response):
        species_list = fetch_data.fetch_top_species(taxon_id=47157, target_num=2, min_obs=100)
        assert len(species_list) == 2
        assert species_list[0]["name"] == "Danaus plexippus"
        assert species_list[1]["name"] == "Vanessa cardui"


def test_download_single_image_success(tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"fake_image_bytes"

    img_path = tmp_path / "test_image.jpg"
    headers = {"User-Agent": "test"}

    with patch("requests.get", return_value=mock_response):
        success = fetch_data.download_single_image("http://example.com/test.jpg", img_path, headers)
        assert success is True
        assert img_path.exists()
        assert img_path.read_bytes() == b"fake_image_bytes"


def test_download_single_image_already_exists(tmp_path):
    img_path = tmp_path / "existing.jpg"
    img_path.write_bytes(b"existing_bytes")

    headers = {"User-Agent": "test"}
    with patch("requests.get") as mock_get:
        success = fetch_data.download_single_image("http://example.com/test.jpg", img_path, headers)
        assert success is True
        mock_get.assert_not_called()
