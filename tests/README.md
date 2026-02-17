# Git-Query Recommender System - Tests

This directory contains tests for the recommendation system.

## Test Philosophy

Since the recommender system is **not yet trained** and we **don't have real data**, our tests focus on:

- ✅ **Functional Testing** - Verify code structure and basic functionality
- ✅ **Contract Testing** - Ensure API contracts are correct  
- ✅ **Validation Testing** - Test input/output validation
- ✅ **Import Testing** - Verify all modules load without errors

We **do NOT test** (until we have trained models and data):
- ❌ Recommendation quality or accuracy
- ❌ Real database interactions
- ❌ Actual embedding/reranking performance

## Test Files

### `test_smoke.py`
Quick sanity checks that core components exist and can be imported.

### `test_recommender.py`
- **Model Validation**: Tests Pydantic models validate correctly
- **Engine Structure**: Tests that all engines can be instantiated
- **Service Structure**: Tests that services can be created

### `test_recommender_api.py`
- **Health Checks**: Basic endpoint availability
- **Request Validation**: API validates inputs correctly
- **Response Structure**: API returns correct response formats
- **Import Tests**: All modules load properly

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_recommender.py -v

# Run with coverage
pytest tests/ --cov=src.recommender

# Run smoke tests only (fastest)
pytest tests/test_smoke.py -v

# Windows batch files
test.bat              # Quick test
run_tests.bat         # Full suite
```

## Current Status

✅ **All functional tests passing**  
⚠️ Some deprecation warnings (non-critical)

## Future Testing (When Data Available)

Once we have:
1. Trained embedding/reranking models
2. Real repository data in the database
3. User interaction history

We can add:
- Recommendation quality tests
- Performance benchmarks
- End-to-end integration tests
- A/B testing validation
- Model accuracy metrics

For now, the tests ensure the **system structure is correct** and **ready for integration**.
