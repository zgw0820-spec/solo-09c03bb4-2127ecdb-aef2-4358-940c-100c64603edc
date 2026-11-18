"""维护发布核验状态机的单元测试（不依赖网络，全部注入假依赖）。

覆盖：
- 两源核验通过才“已发布”，区间必须完整落入各自目标容差；
- 核验不符 / 探测异常 / 下发失败 / 回读不符 → 待处理并给出各源实况；
- 提交即撤销旧发布；变更期间迟到的探测结果不得恢复“已发布”；
- 并发发布：旧尝试的结果不得覆盖新状态；
- 服务重启：状态不残留，必须重新核验；
- 外部改动源端配置后，读取状态时撤销旧结论。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from diag.app.probe import ProbeError, ProbeResult
from diag.app.release import ReleaseManager, SourceTarget


@dataclass
class FakeSource:
    sid: int
    udp_host: str = "127.0.0.1"
    udp_port: int = 10000
    name: str = ""
    config: dict = field(default_factory=lambda: {"offset_ms": 0.0, "reply_delay_ms": 20.0})
    apply_error: Exception | None = None
    readback_error: Exception | None = None


def _target(sid: int, offset=5.0, delay=10.0, tol=2.0) -> SourceTarget:
    return SourceTarget(
        id=sid, name=f"ntp-source-{sid}",
        offset_ms=offset, reply_delay_ms=delay, tolerance_ms=tol,
    )


def _probe_result(offset_ms: float, delay_ms: float) -> ProbeResult:
    """构造一次探测结果：真实钟差=offset_ms，区间半宽=delay_ms/2。"""
    off = offset_ms / 1000.0
    half = delay_ms / 2000.0
    return ProbeResult(
        t1=1.0, t2=1.0 + off, t3=1.0 + off, t4=1.001,
        offset_s=off, delay_s=delay_ms / 1000.0,
        low_s=off - half, high_s=off + half,
    )


class Harness:
    def __init__(self, probe_map=None):
        self.sources = {1: FakeSource(1, name="ntp-source-1"),
                        2: FakeSource(2, name="ntp-source-2")}
        self.sources[1].udp_port = 1234
        self.sources[2].udp_port = 1235
        self.probe_map = probe_map or {}       # sid -> ProbeResult 或 Exception
        self.apply_calls: list[tuple[int, dict]] = []
        self.probe_calls: list[tuple[int, float]] = []
        self.apply_gate: asyncio.Event | None = None

    async def apply(self, src: FakeSource, payload: dict) -> dict:
        self.apply_calls.append((src.sid, dict(payload)))
        if self.apply_gate is not None:
            await self.apply_gate.wait()
        if src.apply_error is not None:
            raise src.apply_error
        src.config = dict(payload)
        return dict(src.config)

    async def fetch(self, src: FakeSource) -> dict:
        if src.readback_error is not None:
            raise src.readback_error
        return dict(src.config)

    def probe(self, host: str, port: int, timeout_s: float) -> ProbeResult:
        # 找出该端口对应源
        sid = next(s.sid for s in self.sources.values() if s.udp_port == port)
        self.probe_calls.append((sid, timeout_s))
        item = self.probe_map.get(sid)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item()
        return item if item is not None else _probe_result(0.0, 0.0)

    def manager(self) -> ReleaseManager:
        return ReleaseManager(self.sources, apply_config=self.apply,
                              fetch_config=self.fetch, probe=self.probe)


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------ 成功路径


def test_release_published_when_both_sources_verified():
    h = Harness({1: _probe_result(5.0, 2.0), 2: _probe_result(3.0, 2.0)})
    mgr = h.manager()

    result = _run(mgr.submit([_target(1, 5.0, 10.0, 2.0), _target(2, 3.0, 12.0, 2.0)], 2000))

    assert result["attempt"]["status"] == "released"
    assert result["attempt"]["published"] is True
    assert all(s["verified"] and s["within_tolerance"] for s in result["attempt"]["sources"])
    assert mgr.snapshot()["published"] is True
    # 下发的是各源自己的目标参数
    assert h.apply_calls == [
        (1, {"offset_ms": 5.0, "reply_delay_ms": 10.0}),
        (2, {"offset_ms": 3.0, "reply_delay_ms": 12.0}),
    ]


def test_interval_must_be_fully_inside_target_tolerance():
    # 区间 [4, 6]，目标 5±2 → [3, 7] 内 → 通过
    h = Harness({1: _probe_result(5.0, 2.0), 2: _probe_result(5.0, 2.0)})
    assert _run(h.manager().submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))[
        "attempt"]["published"] is True

    # 区间 [3.5, 6.5]，目标 5±1 → [4, 6]：上界越界 → 不符
    h = Harness({1: _probe_result(5.0, 3.0), 2: _probe_result(5.0, 3.0)})
    attempt = _run(h.manager().submit([_target(1, 5, 10, 1), _target(2, 5, 10, 1)], 1000))["attempt"]
    assert attempt["published"] is False
    assert all(s["status"] == "mismatch" for s in attempt["sources"])
    assert "未完全落入" in attempt["sources"][0]["error"]


def test_tolerance_boundary_touching_counts_as_verified():
    # 区间 [4, 6]，参数 5±1：端点贴边 → 视为完整落入
    h = Harness({1: _probe_result(5.0, 2.0), 2: _probe_result(5.0, 2.0)})
    attempt = _run(h.manager().submit([_target(1, 5, 10, 1), _target(2, 5, 10, 1)], 1000))["attempt"]
    assert attempt["published"] is True


# ------------------------------------------------------------ 失败路径


def test_mismatch_keeps_pending_with_per_source_live_state():
    h = Harness({1: _probe_result(5.0, 1.0), 2: _probe_result(50.0, 1.0)})
    mgr = h.manager()
    attempt = _run(mgr.submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))["attempt"]

    assert attempt["status"] == "pending" and attempt["published"] is False
    s1, s2 = attempt["sources"]
    assert s1["verified"] is True
    assert s2["verified"] is False and s2["status"] == "mismatch"
    # 实况：目标、回读配置、四时间戳、θ/δ/区间、是否落入容差
    assert s2["target_offset_ms"] == 5.0 and s2["target_interval_ms"] == [3.0, 7.0]
    assert s2["actual_config"] == {"offset_ms": 5.0, "reply_delay_ms": 10.0}
    assert abs(s2["offset_ms"] - 50.0) < 1e-6
    assert s2["timestamps"]["t1"] == 1.0
    assert s2["within_tolerance"] is False
    assert mgr.snapshot()["published"] is False
    assert "源 2" in attempt["reason"]


def test_probe_error_keeps_pending_and_reports_other_source():
    h = Harness({1: _probe_result(5.0, 1.0), 2: ProbeError("回包超时（>1000 ms）")})
    attempt = _run(h.manager().submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))["attempt"]
    assert attempt["published"] is False
    s1, s2 = attempt["sources"]
    assert s1["status"] == "verified"
    assert s2["status"] == "probe_failed" and "超时" in s2["error"]
    # 探测失败也要保留源端实况，便于维护员判断
    assert s2["actual_config"]["offset_ms"] == 5.0


def test_apply_failure_marks_other_source_blocked():
    h = Harness({1: _probe_result(5.0, 1.0), 2: _probe_result(5.0, 1.0)})
    h.sources[2].apply_error = RuntimeError("控制口不可达")
    attempt = _run(h.manager().submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))["attempt"]

    assert attempt["published"] is False
    s1, s2 = attempt["sources"]
    assert s2["status"] == "apply_failed" and "控制口不可达" in s2["error"]
    # 下发失败后仍尽力回读实况（这里是源端未被改动的原配置），便于排查
    assert s2["actual_config"] == {"offset_ms": 0.0, "reply_delay_ms": 20.0}
    assert s1["status"] == "blocked" and s1["verified"] is False
    # 有源下发失败时不做探测（避免用未下发的配置去核验）
    assert h.probe_calls == []


def test_readback_mismatch_is_apply_failure():
    h = Harness({1: _probe_result(5.0, 1.0), 2: _probe_result(5.0, 1.0)})

    async def fetch_lying(src):
        return {"offset_ms": 999.0, "reply_delay_ms": 10.0}

    mgr = ReleaseManager(h.sources, apply_config=h.apply, fetch_config=fetch_lying,
                         probe=h.probe)
    attempt = _run(mgr.submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))["attempt"]
    assert attempt["published"] is False
    assert all(s["status"] == "apply_failed" for s in attempt["sources"])
    assert "回读配置与目标不符" in attempt["sources"][0]["error"]


# ------------------------------------------------------------ 撤销与迟到结果


def test_submit_revokes_previous_published_immediately():
    h = Harness({1: _probe_result(5.0, 1.0), 2: _probe_result(5.0, 1.0)})
    mgr = h.manager()
    _run(mgr.submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))
    assert mgr.snapshot()["published"] is True

    async def scenario():
        h.apply_gate = asyncio.Event()
        task = asyncio.create_task(
            mgr.submit([_target(1, 6, 10, 2), _target(2, 6, 10, 2)], 1000))
        await asyncio.sleep(0.02)  # 让 submit 进入下发阶段（被闸门挡住）
        mid = mgr.snapshot()
        h.apply_gate.set()
        await task
        return mid

    mid = _run(scenario())
    # 变更一开始就撤销旧结论，而不是等新核验结束
    assert mid["published"] is False
    assert mid["status"] == "in_progress"
    assert all(s["verified"] is False for s in mid["sources"])
    # 旧的“已全部核实”实况已被清空，只保留新一轮的下发进度
    assert all(s["status"] in ("pending", "applying") for s in mid["sources"])


def test_late_probe_result_cannot_restore_published():
    """探测进行中发生配置变更：旧尝试的迟到结果必须作废。"""
    release_probe = asyncio.Event()
    h = Harness()
    mgr = h.manager()
    real_probe_one = mgr._probe_one

    async def slow_probe(src, report, timeout_s):
        # 模拟探测很慢，期间发生外部变更
        await release_probe.wait()
        await real_probe_one(src, report, timeout_s)

    h.probe_map = {1: _probe_result(5.0, 1.0), 2: _probe_result(5.0, 1.0)}
    mgr._probe_one = slow_probe  # type: ignore[method-assign]

    async def scenario():
        task = asyncio.create_task(
            mgr.submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))
        await asyncio.sleep(0.05)              # 已下发，卡在探测
        mgr.invalidate("维护员改动了源 1 参数")  # 变更开始，撤销
        release_probe.set()                    # 迟到的探测结果此刻才返回
        attempt = (await task)["attempt"]
        return attempt, mgr.snapshot()

    attempt, state = _run(scenario())
    assert attempt["status"] == "superseded"
    assert attempt["published"] is False
    assert all(s["status"] == "superseded" for s in attempt["sources"])
    assert state["published"] is False
    assert state["status"] == "pending"


def test_slow_old_release_does_not_overwrite_new_release():
    """两个发布交错：旧尝试收尾时 generation 已前进，结论必须丢弃。"""
    slow_gate = asyncio.Event()
    h = Harness()
    h.probe_map = {1: _probe_result(5.0, 1.0), 2: _probe_result(5.0, 1.0)}
    mgr = h.manager()
    real_probe_one = mgr._probe_one
    call_count = {"n": 0}

    async def gated_probe(src, report, timeout_s):
        call_count["n"] += 1
        if call_count["n"] <= 2:        # 第一次发布的探测很慢
            await slow_gate.wait()
        await real_probe_one(src, report, timeout_s)

    mgr._probe_one = gated_probe  # type: ignore[method-assign]

    async def scenario():
        old = asyncio.create_task(
            mgr.submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))
        await asyncio.sleep(0.05)
        h.probe_map = {1: _probe_result(7.0, 1.0), 2: _probe_result(7.0, 1.0)}
        new = (await mgr.submit([_target(1, 7, 10, 2), _target(2, 7, 10, 2)], 1000))["attempt"]
        slow_gate.set()
        old_attempt = (await old)["attempt"]
        return new, old_attempt, mgr.snapshot()

    new, old_attempt, state = _run(scenario())
    assert new["published"] is True
    assert old_attempt["status"] == "superseded"
    # 旧尝试迟到的结论没有覆盖新状态
    assert state["published"] is True
    assert state["generation"] == new["generation"]
    assert all(s["target_offset_ms"] == 7.0 for s in state["sources"])


# ------------------------------------------------------------ 单源调参 / 重启 / 外部改动


def test_single_source_config_change_revokes_release():
    h = Harness({1: _probe_result(5.0, 1.0), 2: _probe_result(5.0, 1.0)})
    mgr = h.manager()
    _run(mgr.submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))
    assert mgr.snapshot()["published"] is True

    _run(mgr.apply_source_config(1, {"offset_ms": 42.0}))
    state = mgr.snapshot()
    assert state["published"] is False
    assert state["status"] == "pending"
    assert all(s["status"] == "config_changed" for s in state["sources"])


def test_restart_never_reports_published():
    """状态只存内存：新进程的管理器必须从“未发布”开始。"""
    h = Harness({1: _probe_result(5.0, 1.0), 2: _probe_result(5.0, 1.0)})
    _run(h.manager().submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))

    restarted = h.manager()
    state = _run(restarted.revalidate())
    assert state["published"] is False
    assert state["status"] == "not_released"
    assert state["sources"] == []


def test_revalidate_detects_external_config_drift():
    h = Harness({1: _probe_result(5.0, 1.0), 2: _probe_result(5.0, 1.0)})
    mgr = h.manager()
    _run(mgr.submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))
    assert mgr.snapshot()["published"] is True

    h.sources[2].config = {"offset_ms": 5.0, "reply_delay_ms": 99.0}  # 绕过服务直接改源端
    state = _run(mgr.revalidate())
    assert state["published"] is False
    assert "源 2" in state["reason"]
    by_id = {s["id"]: s for s in state["sources"]}
    assert by_id[2]["actual_config"]["reply_delay_ms"] == 99.0


def test_pending_source_verification_revoked_by_drift():
    """一次发布只有源 1 通过时，源 1 的“已核实”同样不能残留。"""
    h = Harness({1: _probe_result(5.0, 1.0), 2: _probe_result(50.0, 1.0)})
    mgr = h.manager()
    state = _run(mgr.submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))
    assert state["attempt"]["published"] is False
    assert mgr.snapshot()["sources"][0]["verified"] is True

    h.sources[1].config = {"offset_ms": 5.0, "reply_delay_ms": 33.0}
    state = _run(mgr.revalidate())
    assert state["published"] is False
    s1 = state["sources"][0]
    assert s1["verified"] is False and s1["status"] == "config_changed"
    assert s1["actual_config"]["reply_delay_ms"] == 33.0


def test_revalidate_revokes_when_source_unreachable():
    h = Harness({1: _probe_result(5.0, 1.0), 2: _probe_result(5.0, 1.0)})
    mgr = h.manager()
    _run(mgr.submit([_target(1, 5, 10, 2), _target(2, 5, 10, 2)], 1000))

    h.sources[1].readback_error = RuntimeError("连接被拒绝")
    state = _run(mgr.revalidate())
    assert state["published"] is False
    assert "无法确认" in state["reason"]


def test_release_within_tolerance_after_concurrent_probes():
    """两源探测并发完成顺序不定，结论只由各自结果决定。"""
    h = Harness({1: _probe_result(5.0, 1.0), 2: _probe_result(5.0, 1.0)})
    mgr = h.manager()
    attempt = _run(mgr.submit([_target(1, 5, 10, 2), _target(2, 3, 10, 2)], 1000))["attempt"]
    assert attempt["published"] is False   # 源 2 目标 3ms，实际 5ms
    assert attempt["sources"][0]["status"] == "verified"
    assert attempt["sources"][1]["status"] == "mismatch"
