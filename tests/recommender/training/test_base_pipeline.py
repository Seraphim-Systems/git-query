"""Unit tests for BasePipeline — London School TDD."""

import pytest


# ---------------------------------------------------------------------------
# Concrete test subclass
# ---------------------------------------------------------------------------


def _make_pipeline(
    evaluate_raises=None,
    register_raises=None,
    train_result=None,
):
    from src.recommender.training.pipelines.base_pipeline import BasePipeline

    class ConcretePipeline(BasePipeline):
        call_log = []

        async def fetch(self):
            self.call_log.append("fetch")
            return {"data": "ok"}

        async def train(self, td):
            self.call_log.append("train")
            return train_result or {"model_path": "/m/test.pkl"}

        async def evaluate(self, td, m):
            self.call_log.append("evaluate")
            if evaluate_raises:
                raise evaluate_raises
            return {**m, "ndcg": 0.8}

        async def register(self, m):
            self.call_log.append("register")
            if register_raises:
                raise register_raises

    p = ConcretePipeline()
    p.call_log = []
    return p


# ===========================================================================
# Abstract enforcement
# ===========================================================================


class TestBasePipelineAbstract:
    def test_cannot_instantiate_base_directly(self):
        from src.recommender.training.pipelines.base_pipeline import BasePipeline

        with pytest.raises(TypeError):
            BasePipeline()

    def test_concrete_subclass_can_be_instantiated(self):
        p = _make_pipeline()
        assert p is not None


# ===========================================================================
# run() — phase ordering
# ===========================================================================


class TestBasePipelineRun:
    async def test_run_calls_all_four_phases(self):
        p = _make_pipeline()
        await p.run()
        assert set(p.call_log) == {"fetch", "train", "evaluate", "register"}

    async def test_run_phases_in_order(self):
        p = _make_pipeline()
        await p.run()
        assert p.call_log == ["fetch", "train", "evaluate", "register"]

    async def test_run_returns_metrics_dict(self):
        p = _make_pipeline(train_result={"model_path": "/m/model.pkl"})
        result = await p.run()
        assert isinstance(result, dict)
        assert "model_path" in result

    async def test_run_returns_train_result_when_evaluate_raises(self):
        p = _make_pipeline(evaluate_raises=RuntimeError("eval failed"))
        result = await p.run()
        assert result is not None

    async def test_evaluate_raises_register_still_called(self):
        p = _make_pipeline(evaluate_raises=RuntimeError("eval failed"))
        await p.run()
        assert "register" in p.call_log

    async def test_register_raises_does_not_propagate(self):
        p = _make_pipeline(register_raises=RuntimeError("registry down"))
        result = await p.run()
        assert result is not None

    async def test_evaluate_raises_does_not_propagate(self):
        p = _make_pipeline(evaluate_raises=ValueError("no data"))
        result = await p.run()
        assert result is not None
