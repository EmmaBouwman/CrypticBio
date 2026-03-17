from src.data_gather import check_exists_dir


def test_check_exists_dir_creates_new_folder(tmp_path):
    """Test that a directory is created if it does not exist."""
    test_dir = tmp_path / "new_folder"
    check_exists_dir(test_dir)

    assert test_dir.exists()
    assert test_dir.is_dir()


def test_check_exists_dir_handles_existing_folder(tmp_path):
    """Test that the function doesn't crash if the folder already exists."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    check_exists_dir(test_dir)

    assert test_dir.exists()


def test_check_exists_dir_creates_parents(tmp_path):
    """Test that it creates nested parent directories (parents=True)."""
    test_dir = tmp_path / "test" / "test" / "test"
    check_exists_dir(test_dir)

    assert test_dir.exists()
