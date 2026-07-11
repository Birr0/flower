# Running unit tests 

Full suite:
bash
python -m pytest tests/ -v


With coverage report:
bash
python -m pytest tests/ -v --cov=src/flower --cov-report=term-missing


Specific test file:
bash
python -m pytest tests/test_vae.py -v
python -m pytest tests/test_flow_matching.py -v


Specific test class:
bash
python -m pytest tests/test_models_modules.py::TestWrappedModel -v


Specific test function:
bash
python -m pytest tests/test_models_modules.py::TestWrappedModel::test_cfg_scale_gt1_guidance -v


Skip slow tests (if you add markers later):
bash
python -m pytest tests/ -v -m "not slow"


Verbose output + tracebacks on failure:
bash
python -m pytest tests/ -v --tb=short