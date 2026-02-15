# Test Strategy for Git-Query Recommender System

## Philosophy

Since the recommender system is not yet trained and we don't have real data, our tests focus on:

✅ **Functional Testing** - Verify code structure and basic functionality  
✅ **Contract Testing** - Ensure API contracts are correct  
✅ **Validation Testing** - Test input/output validation  
✅ **Import Testing** - Verify modules can be loaded  

❌ **NOT Testing** (until we have data/models):
- Recommendation quality
- Model accuracy
- Real database interactions
- Actual embedding/reranking performance

## Test Structure

### `test_recommender.py`
- **Model Validation**: Tests that Pydantic models validate correctly
- **Engine Structure**: Tests that engines can be instantiated
- **Service Structure**: Tests that services can be imported and created

### `test_recommender_api.py`
- **Health Checks**: Basic endpoint availability
- **Request Validation**: API validates inputs correctly
- **Response Structure**: API returns correct response formats
- **Import Tests**: All modules load without errors

### `test_smoke.py`
- Quick validation that core components exist

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_recommender.py -v

# Run with coverage
pytest tests/ --cov=src.recommender

# Run smoke tests only
pytest tests/test_smoke.py -v
```

## What to Test Later (When Data is Available)

1. **Recommendation Quality**
   - Test with real repository data
   - Validate ranking algorithms
   - Test personalization accuracy

2. **Performance**
   - Response time benchmarks
   - Load testing
   - Caching effectiveness

3. **Integration**
   - End-to-end recommendation flow
   - Database interactions
   - A/B testing logic

4. **Model Quality**
   - Embedding quality metrics
   - Reranking effectiveness
   - Personalization improvements

## Current Test Status

✅ **5 tests passing** - Basic functionality verified  
⚠️ **Warnings** - Deprecation warnings (datetime.utcnow) - non-critical  

The system is ready for development and integration, even without trained models or data!

