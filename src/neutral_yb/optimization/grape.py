from __future__ import annotations

from typing import Any


class ClosedSystemGRAPE:
    """Unified base and factory namespace for state-vector GRAPE optimizers.

    Closed-system here includes effective non-Hermitian no-jump optimizers whose
    GRAPE loop still propagates state vectors with matrix exponentials.
    """

    @classmethod
    def global_phase(cls, model: Any, config: Any):
        from neutral_yb.optimization.global_phase_grape import _GlobalPhaseClosedSystemGRAPE

        return _GlobalPhaseClosedSystemGRAPE(model, config)

    @classmethod
    def linear_control(cls, model: Any, config: Any):
        from neutral_yb.optimization.linear_control_grape import _LinearControlClosedSystemGRAPE

        return _LinearControlClosedSystemGRAPE(model, config)

    @classmethod
    def amplitude_phase(cls, model: Any, config: Any):
        from neutral_yb.optimization.amplitude_phase_grape import _AmplitudePhaseClosedSystemGRAPE

        return _AmplitudePhaseClosedSystemGRAPE(model, config)

    @classmethod
    def evered_parameterized(cls, *, model: Any, omega_t_over_2pi: float, config: Any):
        from neutral_yb.optimization.evered2023_parameterized_grape import (
            _Evered2023ParameterizedClosedSystemGRAPE,
        )

        return _Evered2023ParameterizedClosedSystemGRAPE(
            model=model,
            omega_t_over_2pi=omega_t_over_2pi,
            config=config,
        )

    @classmethod
    def evered_detuning(cls, *, model: Any, omega_t_over_2pi: float, config: Any):
        from neutral_yb.optimization.evered2023_parameterized_grape import (
            _Evered2023DetuningClosedSystemGRAPE,
        )

        return _Evered2023DetuningClosedSystemGRAPE(
            model=model,
            omega_t_over_2pi=omega_t_over_2pi,
            config=config,
        )

    @classmethod
    def shelved_cr_phase(cls, config: Any, *, include_rydberg_decay: bool = False):
        from neutral_yb.optimization.shelved_cr_phase_grape import (
            _RydbergDecayShelvedCRPhaseClosedSystemGRAPE,
            _ShelvedCRPhaseClosedSystemGRAPE,
        )

        if include_rydberg_decay:
            return _RydbergDecayShelvedCRPhaseClosedSystemGRAPE(config)
        return _ShelvedCRPhaseClosedSystemGRAPE(config)

    @classmethod
    def ma2023_six_level_phase(cls, *, model: Any, config: Any, envelope: Any):
        from neutral_yb.optimization.ma2023_six_level_grape import _Ma2023SixLevelPhaseClosedSystemGRAPE

        return _Ma2023SixLevelPhaseClosedSystemGRAPE(model=model, config=config, envelope=envelope)

    @classmethod
    def ma2023_six_level_chebyshev(cls, *, model: Any, config: Any, envelope: Any):
        from neutral_yb.optimization.ma2023_six_level_grape import (
            _Ma2023SixLevelChebyshevClosedSystemGRAPE,
        )

        return _Ma2023SixLevelChebyshevClosedSystemGRAPE(model=model, config=config, envelope=envelope)


from neutral_yb.optimization.open_system_grape import OpenSystemGRAPE  # noqa: E402
