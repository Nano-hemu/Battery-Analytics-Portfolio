from pathlib import Path
import py_compile


def test_streamlit_app_compiles():
    app_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "streamlit_app.py"
    )

    assert app_path.exists()

    py_compile.compile(
        str(app_path),
        doraise=True,
    )