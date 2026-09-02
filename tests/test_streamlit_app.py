from pathlib import Path
import py_compile


def test_dashboard_modules_compile():
    app_dir = Path(__file__).resolve().parents[1] / "app"

    modules = [
        app_dir / "streamlit_app.py",
        app_dir / "dashboard_utils.py",
    ]

    for module in modules:
        assert module.exists(), f"Missing dashboard module: {module}"
        py_compile.compile(str(module), doraise=True)