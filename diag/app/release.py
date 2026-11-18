"""维护发布（maintenance release）核验状态机。

维护员为每台源填写目标钟差、目标回包延迟与核验容差，提交一次维护发布：

1. 提交的瞬间就撤销旧的“已发布”结论（generation 前进）；
2. 下发两源配置并回读，确认源端确实应用了目标参数；
3. 用真实 UDP 探测，要求每源的钟差不确定区间 [θ-δ/2, θ+δ/2]
   **完整**落在 [目标钟差-容差, 目标钟差+容差] 内；
4. 两源都核实、且核验前后配置未再变化（generation 未变、回读仍等于
   目标值），才置为“已发布”。

并发与迟到结果：
- 每一次配置变更（新的维护发布、或单源调参）都会使 generation 前进；
- 一次尝试在写回最终结论前必须确认 generation 仍是自己提交时的值，
  否则标记为 superseded 并丢弃结论，绝不用迟到的探测结果恢复旧发布；
- 下发阶段用同一把锁串行化，避免两次发布的配置写入交错。

状态仅保存在进程内：服务重启后没有任何“已发布”结论，必须重新核验。
"""
from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .probe import ProbeResult

# 浮点边界容差（毫秒）：配置回读、区间与容差比较
_EPS = 1e-6

ApplyConfig = Callable[[Any, dict], Awaitable[dict]]
FetchConfig = Callable[[Any], Awaitable[dict]]
Probe = Callable[[str, int, float], ProbeResult]


@dataclass(frozen=True)
class SourceTarget:
    """维护员为一台源填写的目标参数。"""

    id: int
    name: str
    offset_ms: float
    reply_delay_ms: float
    tolerance_ms: float

    @property
    def low_ms(self) -> float:
        return self.offset_ms - self.tolerance_ms

    @property
    def high_ms(self) -> float:
        return self.offset_ms + self.tolerance_ms

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "target_offset_ms": self.offset_ms,
            "target_reply_delay_ms": self.reply_delay_ms,
            "target_tolerance_ms": self.tolerance_ms,
            "target_interval_ms": [self.low_ms, self.high_ms],
        }


@dataclass
class SourceReport:
    """单台源在一次发布尝试中的实况与核验结果。"""

    target: SourceTarget
    status: str = "pending"
    verified: bool = False
    error: str | None = None
    applied_config: dict | None = None
    actual_config: dict | None = None
    timestamps: dict | None = None
    offset_ms: float | None = None
    delay_ms: float | None = None
    interval_ms: list[float] | None = None
    within_tolerance: bool | None = None

    def to_dict(self) -> dict:
        data = self.target.to_dict()
        data.update({
            "status": self.status,
            "verified": self.verified,
            "error": self.error,
            "applied_config": copy.deepcopy(self.applied_config),
            "actual_config": copy.deepcopy(self.actual_config),
            "timestamps": copy.deepcopy(self.timestamps),
            "offset_ms": self.offset_ms,
            "delay_ms": self.delay_ms,
            "interval_ms": list(self.interval_ms) if self.interval_ms else None,
            "within_tolerance": self.within_tolerance,
        })
        return data


def _config_matches(actual: dict | None, target: SourceTarget) -> bool:
    if not isinstance(actual, dict):
        return False
    try:
        offset = float(actual["offset_ms"])
        delay = float(actual["reply_delay_ms"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        abs(offset - target.offset_ms) <= _EPS
        and abs(delay - target.reply_delay_ms) <= _EPS
    )


class ReleaseManager:
    """维护发布状态机；依赖以可注入函数提供，便于单元测试。"""

    def __init__(
        self,
        sources: dict[int, Any],
        *,
        apply_config: ApplyConfig,
        fetch_config: FetchConfig,
        probe: Probe,
    ) -> None:
        self._sources = sources
        self._apply_config = apply_config
        self._fetch_config = fetch_config
        self._probe = probe
        self._generation = 0
        self._apply_lock = asyncio.Lock()
        self._state: dict = self._blank_state()

    # ------------------------------------------------------------ 状态

    def _blank_state(self) -> dict:
        return {
            "generation": self._generation,
            "status": "not_released",
            "published": False,
            "reason": "尚未提交维护发布（服务重启后需重新核验）",
            "updated_at": time.time(),
            "timeout_ms": None,
            "sources": [],
        }

    def snapshot(self) -> dict:
        return copy.deepcopy(self._state)

    def invalidate(self, reason: str, source_status: str = "config_changed") -> None:
        """任何配置变更开始：立即撤销旧发布结论，并让在途尝试作废。"""
        self._generation += 1
        state = self._state
        had_conclusion = bool(state.get("sources")) or bool(state.get("published"))
        if had_conclusion:
            for rep in state.get("sources", []):
                # 旧的实测都是旧配置下取得的，变更后不再代表当前状态：
                # 结论与实测一并作废，只保留目标与源端回读配置供排查。
                rep["verified"] = False
                rep["status"] = source_status
                rep["error"] = reason
                rep["timestamps"] = None
                rep["offset_ms"] = None
                rep["delay_ms"] = None
                rep["interval_ms"] = None
                rep["within_tolerance"] = None
            state["status"] = "pending"
            state["published"] = False
            state["reason"] = reason
            state["updated_at"] = time.time()
        state["generation"] = self._generation

    async def revalidate(self) -> dict:
        """读取发布状态前，回读两源配置；不能确认未变化则撤销“已发布”。

        防止有人绕过本服务直接改源端控制口：只要有源被标记为“已核实”，
        就回读其配置；与目标不符或控制口不可达（无法确认），立即撤销该
        源结论与整体“已发布”，不再把旧结论展示为已发布。
        """
        state = self._state
        if not any(rep.get("verified") for rep in state.get("sources", [])):
            return self.snapshot()

        generation = self._generation
        targets = [
            SourceTarget(
                id=rep["id"], name=rep["name"],
                offset_ms=rep["target_offset_ms"],
                reply_delay_ms=rep["target_reply_delay_ms"],
                tolerance_ms=rep["target_tolerance_ms"],
            )
            for rep in state.get("sources", [])
            if rep.get("verified")
        ]

        async def check(target: SourceTarget) -> tuple[str | None, dict | None]:
            try:
                actual = await self._fetch_config(self._sources[target.id])
            except Exception as exc:  # noqa: BLE001
                return (f"源 {target.id} 控制口不可达，无法确认配置未变化：{exc}", None)
            if not _config_matches(actual, target):
                return (
                    f"源 {target.id} 配置已变化（回读 offset={actual.get('offset_ms')} ms, "
                    f"reply_delay={actual.get('reply_delay_ms')} ms），旧发布结论已撤销",
                    actual,
                )
            return (None, actual)

        checks = await asyncio.gather(*(check(t) for t in targets))
        problems = [p for p, _ in checks if p]
        if generation != self._generation:  # 期间已有更新的发布/调参
            return self.snapshot()
        if problems:
            self.invalidate("；".join(problems), source_status="config_changed")
            # 把导致撤销的源端实况一并呈现，便于维护员处理
            actual_by_id = {
                t.id: actual for t, (_, actual) in zip(targets, checks) if actual is not None
            }
            for report in self._state.get("sources", []):
                if report["id"] in actual_by_id:
                    report["actual_config"] = actual_by_id[report["id"]]
        return self.snapshot()

    async def apply_source_config(self, sid: int, payload: dict) -> dict:
        """单源实时调参：先撤销发布结论，再与发布下发串行写入。"""
        self.invalidate(f"源 {sid} 配置已变更，旧发布结论已撤销，需重新核验")
        async with self._apply_lock:
            return await self._apply_config(self._sources[sid], payload)

    # ------------------------------------------------------------ 发布

    async def submit(self, targets: list[SourceTarget], timeout_ms: int) -> dict:
        """提交一次维护发布，等待核验结束后返回 {release, attempt}。"""
        self._generation += 1
        generation = self._generation
        timeout_s = timeout_ms / 1000.0
        targets_by_id = {t.id: t for t in targets}
        reports = [
            SourceReport(target=targets_by_id[sid]) for sid in self._sources
        ]

        # 提交即撤销旧结论：在任何 await 之前完成。
        self._state = {
            "generation": generation,
            "status": "in_progress",
            "published": False,
            "reason": "维护发布核验进行中：下发配置并用真实 UDP 回包核验两源",
            "updated_at": time.time(),
            "timeout_ms": timeout_ms,
            "sources": [r.to_dict() for r in reports],
        }

        # ---- 1) 下发配置 + 回读（与单源调参、其他发布互斥） ----
        async with self._apply_lock:
            if generation != self._generation:
                return self._finish(generation, reports, timeout_ms, superseded=True)

            for report in reports:
                report.status = "applying"
            await asyncio.gather(*(
                self._apply_one(self._sources[report.target.id], report)
                for report in reports
            ))

        if generation != self._generation:
            return self._finish(generation, reports, timeout_ms, superseded=True)

        apply_failed = [r for r in reports if r.status == "apply_failed"]
        if apply_failed:
            for report in reports:
                if report.status != "apply_failed":
                    report.status = "blocked"
                    report.error = (
                        f"源 {apply_failed[0].target.id} 下发失败，"
                        "本轮未对该源进行 UDP 核验"
                    )
            return self._finish(generation, reports, timeout_ms)

        # ---- 2) 真实 UDP 探测 ----
        for report in reports:
            report.status = "probing"
        self._refresh_progress(generation, reports)
        await asyncio.gather(*(
            self._probe_one(self._sources[report.target.id], report, timeout_s)
            for report in reports
        ))

        if generation != self._generation:
            return self._finish(generation, reports, timeout_ms, superseded=True)

        # ---- 3) 核验后回读：配置期间不得再被改动 ----
        await asyncio.gather(*(
            self._recheck_one(self._sources[report.target.id], report)
            for report in reports
        ))

        return self._finish(generation, reports, timeout_ms)

    async def _apply_one(self, src: Any, report: SourceReport) -> None:
        target = report.target
        payload = {
            "offset_ms": target.offset_ms,
            "reply_delay_ms": target.reply_delay_ms,
        }
        try:
            applied = await self._apply_config(src, payload)
            report.applied_config = applied
            actual = await self._fetch_config(src)
            report.actual_config = actual
        except Exception as exc:  # noqa: BLE001 - 任何下发/回读异常都算该源失败
            report.status = "apply_failed"
            report.verified = False
            report.error = f"配置下发失败：{exc}"
            # 尽力回读一次，尽量给维护员展示该源当前实况（失败也不掩盖原因）
            try:
                report.actual_config = await self._fetch_config(src)
            except Exception:  # noqa: BLE001
                pass
            return
        if not _config_matches(actual, target):
            report.status = "apply_failed"
            report.error = (
                "源端回读配置与目标不符："
                f"offset={actual.get('offset_ms')} ms, "
                f"reply_delay={actual.get('reply_delay_ms')} ms"
            )

    async def _probe_one(self, src: Any, report: SourceReport, timeout_s: float) -> None:
        target = report.target
        try:
            res = await asyncio.to_thread(
                self._probe, src.udp_host, src.udp_port, timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - 超时/无效回包/意外异常都算探测失败
            report.status = "probe_failed"
            report.verified = False
            report.error = f"UDP 探测异常：{exc}"
            return

        interval = [res.low_s * 1000.0, res.high_s * 1000.0]
        report.timestamps = {
            "t1": res.t1, "t2": res.t2, "t3": res.t3, "t4": res.t4,
        }
        report.offset_ms = res.offset_s * 1000.0
        report.delay_ms = res.delay_s * 1000.0
        report.interval_ms = interval
        within = (
            interval[0] >= target.low_ms - _EPS
            and interval[1] <= target.high_ms + _EPS
        )
        report.within_tolerance = within
        if within:
            report.status = "verified"
            report.verified = True
        else:
            report.status = "mismatch"
            report.verified = False
            report.error = (
                f"钟差区间 [{interval[0]:.3f}, {interval[1]:.3f}] ms "
                f"未完全落入目标 {target.offset_ms:+.3f}±{target.tolerance_ms:.3f} ms "
                f"（目标区间 [{target.low_ms:.3f}, {target.high_ms:.3f}] ms）"
            )

    async def _recheck_one(self, src: Any, report: SourceReport) -> None:
        if report.status != "verified":
            return
        try:
            actual = await self._fetch_config(src)
        except Exception as exc:  # noqa: BLE001
            report.status = "config_changed"
            report.verified = False
            report.error = f"核验后无法回读配置，不能确认配置未再变化：{exc}"
            return
        report.actual_config = actual
        if not _config_matches(actual, report.target):
            report.status = "config_changed"
            report.verified = False
            report.error = (
                "核验期间配置已被改动（回读 "
                f"offset={actual.get('offset_ms')} ms, "
                f"reply_delay={actual.get('reply_delay_ms')} ms），本次结论作废"
            )

    # ------------------------------------------------------------ 收尾

    def _refresh_progress(self, generation: int, reports: list[SourceReport]) -> None:
        if generation == self._generation:
            self._state["sources"] = [r.to_dict() for r in reports]

    def _finish(
        self,
        generation: int,
        reports: list[SourceReport],
        timeout_ms: int,
        *,
        superseded: bool = False,
    ) -> dict:
        # 走到收尾时再确认一次：期间若发生任何配置变更，本次尝试一律作废。
        if generation != self._generation:
            superseded = True
        if superseded:
            for report in reports:
                report.status = "superseded"
                report.verified = False
                report.error = "本次发布已被更新的配置变更取代，结果作废"
            status = "superseded"
            published = False
            reason = "本次发布已被更新的配置变更取代，结果作废，以最新状态为准"
        else:
            all_ok = all(r.verified for r in reports)
            if all_ok:
                status = "released"
                published = True
                reason = "两源均已按目标参数核实，且配置未再变化，维护发布已发布"
            else:
                status = "pending"
                published = False
                parts: list[str] = []
                for label, key in (
                    ("下发失败", "apply_failed"),
                    ("探测异常", "probe_failed"),
                    ("核验不符", "mismatch"),
                    ("配置已变", "config_changed"),
                    ("未核验", "blocked"),
                ):
                    ids = [str(r.target.id) for r in reports if r.status == key]
                    if ids:
                        parts.append(f"源 {', '.join(ids)} {label}")
                reason = (
                    f"{'；'.join(parts)}，维护发布保持待处理；"
                    "请按下方各源实况处理后重试"
                )

        attempt = {
            "generation": generation,
            "status": status,
            "published": published,
            "reason": reason,
            "updated_at": time.time(),
            "timeout_ms": timeout_ms,
            "superseded": superseded,
            "sources": [r.to_dict() for r in reports],
        }

        # 关键：只有 generation 仍是本次尝试的值，才允许写回全局结论。
        # 迟到的结果绝不覆盖更新后的状态，更不会把已撤销的发布恢复成“已发布”。
        if generation == self._generation and not superseded:
            self._state = copy.deepcopy(attempt)
        return {"release": self.snapshot(), "attempt": attempt}
